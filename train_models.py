


# -*- coding: utf-8 -*-
"""
Comparison of models to classify FATIGUE / NO FATIGUE using sEMG.

Requires process_emg.py and qc_emg.py in the same folder, and qc_emg.py must have 
been executed beforehand (uses ./QC/dataset_ventanas_completo.csv).

Experimental design
-------------------
* Label: position within each set (first 25% = no fatigue, last 25% = fatigue).
* Validation: Leave-One-Subject-Out (LOSO). The test subject is never seen during 
  training, and scaling is fitted only on the training subjects.
* Hyper-parameters fixed a priori (not tuned by looking at the test subject).
* Exclusion scenarios:
    E0  No exclusion   all windows (includes REVIEW/EXCLUDE files and non-eligible legs)
    E1  Standard QC    OK files + eligible subject-leg units
    E2  Strict QC      E1 without legs with spikes, without evidence of fatigue, or atypical
* Two decision levels: window (250 ms) and contraction (average of its windows).

Outputs in ./Modelos:
    tabla_comparacion.csv/.md   main table (all models and scenarios)
    tabla_por_clase.csv/.md     per-class performance of the best model
    por_sujeto_mejor.csv        per-subject accuracy of the best model
    comparacion_modelos.png, matriz_confusion_mejor.png, por_sujeto_mejor.png
"""


import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             recall_score, precision_recall_fscore_support, confusion_matrix)

from process_emg import RUTA_DATASET, FS, acondicionar
from qc_emg import leer_crudo

warnings.filterwarnings("ignore")
RUTA_QC = "./QC"
RUTA_MOD = "./Modelos"
SEMILLA = 42
ESCENARIO_PRINCIPAL = "E1"   # escenario donde se elige el mejor modelo
# Modelo principal fijado a priori: los tres modelos lineales son equivalentes (diferencias
# <= 0.3 puntos) y la regresión logística da probabilidades calibradas. None = elegir el mejor.
MODELO_PRINCIPAL = "Regresión logística"
N_BASAL = 3                  # contracciones basales para normalizar (igual que qc_emg)
EPOCAS_CNN = 12

CARACT = ["rms_n", "mav_n", "wl_n", "zc_n", "ssc_n", "mnf_n", "mdf_n",
          "fi_nsm5_n", "ratio_bajo_alto_n"]
CLASES = ["No fatiga", "Fatiga"]


# =====================================================================
# DATOS
# =====================================================================
def cargar_datos():
    d = pd.read_csv(os.path.join(RUTA_QC, "dataset_ventanas_completo.csv"))
    for c in ["zc", "ssc"]:   # normalizar ZC y SSC respecto a la basal de cada sujeto-pierna
        base = d[d.idx_global < N_BASAL].groupby(["Sujeto", "Pierna"])[c].median().rename(c + "_b")
        d = d.merge(base, on=["Sujeto", "Pierna"], how="left")
        d[c + "_n"] = d[c] / d[c + "_b"].replace(0, np.nan)
    d = d.reset_index(drop=True)
    d["rms_rel"] = d["rms_n"]                         # sin log, para escalar la señal de la CNN
    for c in CARACT:                                  # razones positivas y asimétricas -> log
        d[c] = np.log(d[c].clip(lower=1e-6))
    d["contr_id"] = d.Archivo + "#" + d.contraccion.astype(str)
    return d


def escenarios(d):
    lab = d.etiqueta.notna() & d[CARACT].notna().all(axis=1)
    espigas = d.groupby(["Sujeto", "Pierna"]).motivos_archivo.transform(
        lambda s: s.fillna("").str.contains("espiga").any())
    e1 = lab & (d.estado_archivo == "OK") & (d.apto_entrenamiento == True)
    e2 = e1 & ~espigas & (d.sin_evidencia_fatiga != True) & d.atipico_en.isna()
    return {"E0": d[lab], "E1": d[e1], "E2": d[e2]}


def ventanas_crudas(d):
    """Señal filtrada de cada ventana (para la CNN), escalada por la RMS basal del sujeto-pierna."""
    cache = os.path.join(RUTA_MOD, "ventanas_crudas.npy")
    if os.path.exists(cache):
        X = np.load(cache)
        if len(X) == len(d):
            return X
    n_w = int(0.250 * FS)
    X = np.zeros((len(d), n_w), np.float32)
    for (suj, arch), g in d.groupby(["Sujeto", "Archivo"]):
        v, _ = leer_crudo(os.path.join(RUTA_DATASET, suj, arch))
        x = acondicionar(v)
        for idx, t0 in zip(g.index, g.t_ventana):
            i = int(round(t0 * FS))
            X[idx] = x[i:i + n_w]
    # cada ventana pasa a tener RMS = rms / rms_basal (conserva el aumento de amplitud)
    X = X / (np.sqrt(np.mean(X ** 2, axis=1, keepdims=True)) + 1e-12) * d.rms_rel.values[:, None]
    np.save(cache, X)
    return X


# =====================================================================
# CNN 1D  (PyTorch si está instalado; si no, implementación NumPy+autograd equivalente)
# =====================================================================
class CNN1D:
    """Conv(16,k9)-ReLU-MaxPool4 → Conv(32,k7)-ReLU-MaxPool4 → Conv(32,k5)-ReLU
       → GlobalAvgPool → Dropout(0.3) → Linear(1).  Entrada: 250 muestras (250 ms)."""

    def __init__(self, epocas=EPOCAS_CNN, lote=64, lr=1e-3, wd=1e-4, dropout=0.3, semilla=SEMILLA):
        self.epocas, self.lote, self.lr, self.wd, self.dropout, self.semilla = \
            epocas, lote, lr, wd, dropout, semilla
        try:
            import torch  # noqa: F401
            self.backend = "torch"
        except Exception:
            self.backend = "numpy"

    # ---------------- PyTorch ----------------
    def _fit_torch(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.semilla)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 9), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 7), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 32, 5), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(self.dropout), nn.Linear(32, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt = torch.tensor(X[:, None, :], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.semilla)
        for _ in range(self.epocas):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=g).split(self.lote):
                opt.zero_grad()
                z = self.net(Xt[idx]).squeeze(1)
                nn.functional.binary_cross_entropy_with_logits(z, yt[idx]).backward()
                opt.step()

    def _score_torch(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(X[:, None, :], dtype=torch.float32)).squeeze(1)
        return torch.sigmoid(z).numpy()

    # ---------------- NumPy + autograd ----------------
    @staticmethod
    def _conv(x, W, b):                       # x: (N, C, L), W: (F, C, K)
        import autograd.numpy as anp
        K = W.shape[2]
        L = x.shape[2] - K + 1
        idx = np.arange(L)[:, None] + np.arange(K)[None, :]
        return anp.einsum("nclk,fck->nfl", x[:, :, idx], W) + b[None, :, None]

    @staticmethod
    def _pool(x, p):
        import autograd.numpy as anp
        L = (x.shape[2] // p) * p
        return anp.max(anp.reshape(x[:, :, :L], (x.shape[0], x.shape[1], L // p, p)), axis=3)

    def _forward(self, P, x, mask=None):
        import autograd.numpy as anp
        h = self._pool(anp.maximum(self._conv(x, P["W1"], P["b1"]), 0), 4)
        h = self._pool(anp.maximum(self._conv(h, P["W2"], P["b2"]), 0), 4)
        h = anp.mean(anp.maximum(self._conv(h, P["W3"], P["b3"]), 0), axis=2)
        if mask is not None:
            h = h * mask
        return anp.dot(h, P["W4"])[:, 0] + P["b4"][0]

    def _fit_numpy(self, X, y):
        import autograd.numpy as anp
        from autograd import grad
        rng = np.random.default_rng(self.semilla)

        def he(*s):
            return rng.normal(0, np.sqrt(2 / np.prod(s[1:])), s)

        P = {"W1": he(16, 1, 9), "b1": np.zeros(16), "W2": he(32, 16, 7), "b2": np.zeros(32),
             "W3": he(32, 32, 5), "b3": np.zeros(32), "W4": rng.normal(0, 0.1, (32, 1)), "b4": np.zeros(1)}

        def perdida(P, xb, yb, mask):
            z = self._forward(P, xb, mask)
            return anp.mean(anp.maximum(z, 0) - z * yb + anp.log1p(anp.exp(-anp.abs(z))))

        g = grad(perdida)
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v2 = {k: np.zeros_like(v) for k, v in P.items()}
        t, b1, b2 = 0, 0.9, 0.999
        X3 = X[:, None, :].astype(np.float64)
        for _ in range(self.epocas):
            orden = rng.permutation(len(X3))
            for s in range(0, len(orden), self.lote):
                i = orden[s:s + self.lote]
                mask = (rng.random((len(i), 32)) > self.dropout) / (1 - self.dropout)
                gr = g(P, X3[i], y[i].astype(float), mask)
                t += 1
                for k in P:
                    P[k] = P[k] * (1 - self.lr * self.wd)          # decaimiento desacoplado (AdamW)
                    m[k] = b1 * m[k] + (1 - b1) * gr[k]
                    v2[k] = b2 * v2[k] + (1 - b2) * gr[k] ** 2
                    P[k] = P[k] - self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v2[k] / (1 - b2 ** t)) + 1e-8)
        self.P = P

    def _score_numpy(self, X):
        z = np.concatenate([self._forward(self.P, X[i:i + 512, None, :].astype(np.float64))
                            for i in range(0, len(X), 512)])
        return 1 / (1 + np.exp(-z))

    # ---------------- interfaz ----------------
    def fit(self, X, y):
        (self._fit_torch if self.backend == "torch" else self._fit_numpy)(X, y)
        return self

    def score(self, X):
        return self._score_torch(X) if self.backend == "torch" else self._score_numpy(X)


# =====================================================================
# MODELOS
# =====================================================================
def modelos():
    esc = StandardScaler
    return {
        "Regresión logística": make_pipeline(esc(), LogisticRegression(C=1.0, max_iter=2000)),
        "LDA": make_pipeline(esc(), LinearDiscriminantAnalysis()),
        "Naive Bayes": make_pipeline(esc(), GaussianNB()),
        "KNN (k=25)": make_pipeline(esc(), KNeighborsClassifier(n_neighbors=25)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                n_jobs=1, random_state=SEMILLA),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            random_state=SEMILLA),
        "SVM lineal": make_pipeline(esc(), LinearSVC(C=0.1, max_iter=5000)),
        "SVM RBF": make_pipeline(esc(), SVC(C=1.0, gamma="scale")),
        "MLP (64-32)": make_pipeline(esc(), MLPClassifier((64, 32), alpha=1e-3, early_stopping=True,
                                                          max_iter=300, random_state=SEMILLA)),
        "CNN 1D (señal cruda)": "cnn",
    }


def puntuacion(m, X):
    """(puntuación continua, umbral de decisión)."""
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1], 0.5
    return m.decision_function(X), 0.0


def metricas(y, s, thr):
    p = (s >= thr).astype(int)
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "f1": f1_score(y, p, average="macro"),
            "auc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
            "sens": recall_score(y, p, pos_label=1, zero_division=0),
            "esp": recall_score(y, p, pos_label=0, zero_division=0)}


def evaluar_loso(plantilla, D, Xraw, cache_folds=None):
    y = D.etiqueta.values.astype(int)
    grupos = D.Sujeto.values
    X = D[CARACT].values
    s_all, thr = np.zeros(len(D)), 0.5
    for tr, te in LeaveOneGroupOut().split(X, y, grupos):
        if isinstance(plantilla, str):
            # la CNN es lenta: se guarda cada pliegue para poder reanudar
            f = os.path.join(cache_folds, grupos[te][0] + ".npy") if cache_folds else None
            if f and os.path.exists(f):
                s = np.load(f)
            else:
                s = CNN1D().fit(Xraw[tr], y[tr]).score(Xraw[te])
                if f:
                    os.makedirs(cache_folds, exist_ok=True)
                    np.save(f, s)
            thr = 0.5
        else:
            m = clone(plantilla).fit(X[tr], y[tr])
            s, thr = puntuacion(m, X[te])
        s_all[te] = s
    pred = pd.DataFrame({"Sujeto": grupos, "contr_id": D.contr_id.values, "y": y, "s": s_all})
    return pred, thr


def resumir(pred, thr):
    """Métricas por sujeto (media ± DE) a nivel ventana y contracción."""
    out = {}
    pc = pred.groupby(["Sujeto", "contr_id"]).agg(y=("y", "first"), s=("s", "mean")).reset_index()
    for nivel, P in [("vent", pred), ("contr", pc)]:
        ps = pd.DataFrame([metricas(g.y.values, g.s.values, thr) for _, g in P.groupby("Sujeto")])
        for k in ps:
            out[f"{nivel}_{k}"] = ps[k].mean()
            out[f"{nivel}_{k}_sd"] = ps[k].std()
    return out, pc


# =====================================================================
# TABLAS Y FIGURAS
# =====================================================================
def fmt(m, sd):
    return f"{100 * m:.1f} ± {100 * sd:.1f}"


def tabla_md(T, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(T.columns) + " |\n|" + "---|" * len(T.columns) + "\n")
        for _, r in T.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")


def figura_comparacion(R, ruta):
    piv = R.pivot(index="Modelo", columns="Escenario", values="vent_acc")
    piv = piv.loc[R[R.Escenario == ESCENARIO_PRINCIPAL].sort_values("vent_acc").Modelo]
    ax = piv.mul(100).plot.barh(figsize=(10, 7), width=0.8)
    ax.set_xlabel("Exactitud LOSO por ventana (%)")
    ax.set_xlim(50, 100)
    ax.grid(axis="x", ls="--", alpha=0.5)
    ax.set_title("Comparación de modelos y escenarios de exclusión")
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)
    plt.close()


def figura_confusion(cm_v, cm_c, nombre, ruta):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, tit in [(axs[0], cm_v, "Por ventana (250 ms)"), (axs[1], cm_c, "Por contracción")]:
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * cmn[i, j]:.1f} %)", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(CLASES)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CLASES)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(tit)
    fig.suptitle(f"Matriz de confusión — {nombre} (LOSO)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, dpi=140)
    plt.close()


# =====================================================================
# PRINCIPAL
# =====================================================================
def main(solo_modelos=None):
    os.makedirs(RUTA_MOD, exist_ok=True)
    d = cargar_datos()
    print("Extrayendo ventanas de señal para la CNN...")
    Xraw_all = ventanas_crudas(d)
    E = escenarios(d)
    for k, D in E.items():
        print(f"{k}: {len(D)} ventanas | {D.Sujeto.nunique()} sujetos | "
              f"{D.groupby(['Sujeto', 'Pierna']).ngroups} sujeto-pierna | "
              f"clases {D.etiqueta.value_counts().to_dict()}")

    filas, predicciones = [], {}
    for esc, D in E.items():
        Xraw = Xraw_all[D.index.values]
        Dr = D.reset_index(drop=True)
        for nombre, plantilla in modelos().items():
            if solo_modelos and nombre not in solo_modelos:
                continue
            t0 = time.time()
            cache = os.path.join(RUTA_MOD, "predicciones",
                                 f"{esc}_{''.join(ch for ch in nombre if ch.isalnum())}.csv")
            if os.path.exists(cache):                 # reanudar sin repetir lo ya calculado
                pred = pd.read_csv(cache)
                thr = 0.0 if nombre.startswith("SVM") else 0.5
            else:
                pred, thr = evaluar_loso(plantilla, Dr, Xraw,
                                         cache_folds=os.path.join(RUTA_MOD, "predicciones", f"cnn_{esc}"))
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                pred.to_csv(cache, index=False)
            res, pc = resumir(pred, thr)
            predicciones[(esc, nombre)] = (pred, pc, thr)
            filas.append({"Escenario": esc, "Modelo": nombre, **res})
            print(f"  {esc} | {nombre:22s} acc={100 * res['vent_acc']:.1f}%  "
                  f"contr={100 * res['contr_acc']:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(filas)
    R.to_csv(os.path.join(RUTA_MOD, "resultados_completos.csv"), index=False, encoding="utf-8-sig")
    R.to_csv("results.csv", index=False, encoding="utf-8-sig")   # exigido por la lista de entrega

    # ---------------- Tabla principal ----------------
    T = pd.DataFrame({
        "Escenario": R.Escenario, "Modelo": R.Modelo,
        "Accuracy (%)": [fmt(a, b) for a, b in zip(R.vent_acc, R.vent_acc_sd)],
        "Bal. accuracy (%)": [fmt(a, b) for a, b in zip(R.vent_bal_acc, R.vent_bal_acc_sd)],
        "F1 macro (%)": [fmt(a, b) for a, b in zip(R.vent_f1, R.vent_f1_sd)],
        "AUC": [f"{a:.3f}" for a in R.vent_auc],
        "Sensibilidad (%)": [f"{100 * a:.1f}" for a in R.vent_sens],
        "Especificidad (%)": [f"{100 * a:.1f}" for a in R.vent_esp],
        "Accuracy por contracción (%)": [fmt(a, b) for a, b in zip(R.contr_acc, R.contr_acc_sd)],
    })
    T.to_csv(os.path.join(RUTA_MOD, "tabla_comparacion.csv"), index=False, encoding="utf-8-sig")
    tabla_md(T, os.path.join(RUTA_MOD, "tabla_comparacion.md"))
    figura_comparacion(R, os.path.join(RUTA_MOD, "comparacion_modelos.png"))

    # ---------------- Mejor modelo: desempeño por clase ----------------
    Rp = R[R.Escenario == ESCENARIO_PRINCIPAL]
    mejor = (MODELO_PRINCIPAL if MODELO_PRINCIPAL in set(Rp.Modelo)
             else Rp.sort_values(["vent_acc", "vent_auc"], ascending=False).iloc[0].Modelo)
    pred, pc, thr = predicciones[(ESCENARIO_PRINCIPAL, mejor)]
    filas_c, cms = [], []
    for nivel, P in [("Ventana", pred), ("Contracción", pc)]:
        yp = (P.s >= thr).astype(int)
        pr, rc, f1, sup = precision_recall_fscore_support(P.y, yp, labels=[0, 1])
        for i, c in enumerate(CLASES):
            filas_c.append({"Nivel": nivel, "Clase": c, "Precisión (%)": f"{100 * pr[i]:.1f}",
                            "Recall (%)": f"{100 * rc[i]:.1f}", "F1 (%)": f"{100 * f1[i]:.1f}",
                            "Soporte": int(sup[i])})
        filas_c.append({"Nivel": nivel, "Clase": "Macro promedio",
                        "Precisión (%)": f"{100 * pr.mean():.1f}", "Recall (%)": f"{100 * rc.mean():.1f}",
                        "F1 (%)": f"{100 * f1.mean():.1f}", "Soporte": int(sup.sum())})
        filas_c.append({"Nivel": nivel, "Clase": "Accuracy global",
                        "Precisión (%)": "", "Recall (%)": "",
                        "F1 (%)": f"{100 * accuracy_score(P.y, yp):.1f}", "Soporte": int(sup.sum())})
        cms.append(confusion_matrix(P.y, yp, labels=[0, 1]))
    TC = pd.DataFrame(filas_c)
    TC.to_csv(os.path.join(RUTA_MOD, "tabla_por_clase.csv"), index=False, encoding="utf-8-sig")
    tabla_md(TC, os.path.join(RUTA_MOD, "tabla_por_clase.md"))
    figura_confusion(cms[0], cms[1], mejor, os.path.join(RUTA_MOD, "matriz_confusion_mejor.png"))

    ps = pd.DataFrame([{"Sujeto": s, "acc_ventana": accuracy_score(g.y, g.s >= thr),
                        "acc_contraccion": accuracy_score(pc[pc.Sujeto == s].y, pc[pc.Sujeto == s].s >= thr)}
                       for s, g in pred.groupby("Sujeto")])
    ps.to_csv(os.path.join(RUTA_MOD, "por_sujeto_mejor.csv"), index=False, encoding="utf-8-sig")
    ax = ps.set_index("Sujeto").mul(100).sort_values("acc_ventana").plot.barh(figsize=(9, 6))
    ax.set_xlabel("Exactitud (%)"); ax.set_xlim(40, 100); ax.grid(axis="x", ls="--", alpha=0.5)
    ax.set_title(f"{mejor} — exactitud por sujeto (LOSO, {ESCENARIO_PRINCIPAL})")
    plt.tight_layout(); plt.savefig(os.path.join(RUTA_MOD, "por_sujeto_mejor.png"), dpi=140); plt.close()

    print(f"\nMejor modelo en {ESCENARIO_PRINCIPAL}: {mejor}")
    print(TC.to_string(index=False))
    return R


if __name__ == "__main__":
    main()