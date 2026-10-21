


# -*- coding: utf-8 -*-
"""
Quality control (QC), exclusion criteria, and windowed dataset
for classifying muscle fatigue using sEMG.

Requires procesar_emg.py in the same folder (reuses filters, envelope,
peak detection, and name parsing to ensure consistency).

Exclusion levels (from highest to lowest):
    1. Subject/leg    -> sufficient sets, homogeneity relative to the group
    2. File           -> integrity, saturation, powerline noise, SNR, rhythm
    3. Contraction    -> duration, SNR, spikes, atypical amplitude, physiological MDF
    4. Window         -> spikes or crest factor within the window

Outputs in ./QC:
    qc_archivos.csv       OK / REVIEW / EXCLUDE status per file and reasons
    qc_contracciones.csv  metrics per contraction and whether it is valid
    qc_sujetos.csv        summary per subject-leg and homogeneity flags
    dataset_ventanas.csv  valid windows with features and candidate labels
    homogeneidad.png      comparison between subjects and progression per set
    revisar/*.png         REVIEW/EXCLUDE files with marked contractions
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from process_emg import (RUTA_DATASET, FS, acondicionar, calcular_envolvente,
                          detectar_global, detectar_adaptativo, limites_contracciones,
                          extraer_pierna, extraer_repeticion, limpiar)

RUTA_QC = "./QC"

# =====================================================================
# CRITERIOS (ajústalos y documenta los valores finales en tu informe)
# =====================================================================
# --- Archivo: integridad -> EXCLUIR
DURACION_MIN_S = 10.0          # registro demasiado corto
NAN_MAX = 0.02                 # fracción de muestras no numéricas
SATURACION_MAX = 0.005         # fracción de muestras pegadas al máx./mín. del ADC
MIN_CONTR_VALIDAS = 4          # contracciones válidas mínimas por archivo
FRAC_CONTR_VALIDAS_MIN = 0.60  # fracción mínima de contracciones válidas

# --- Archivo: calidad -> REVISAR (inspección visual antes de decidir)
TOL_FS = 0.02                  # desviación de la fs estimada respecto a 1000 Hz
SALTO_MAX_MUESTRAS = 2         # salto en el tiempo que cuenta como hueco
PERDIDAS_MAX = 0.01            # fracción de muestras perdidas tolerada
HUECO_MAX_MS = 100.0           # hueco individual máximo tolerado
PLANO_MAX_S = 0.5              # tramo sin cambio de valor (electrodo suelto)
RATIO_RED_MAX = 0.30           # potencia 58-62 Hz / 20-450 Hz (señal sin filtrar)
RATIO_BAJA_MAX = 0.50          # potencia < 20 Hz / total (movimiento / deriva)
SNR_ARCHIVO_MIN_DB = 15.0      # SNR mediana: contracción vs. piso de ruido eléctrico
CONTRASTE_MIN_DB = 6.0         # contracción vs. pausa entre contracciones (relajación)
REPOSO_MIN_S = 1.0             # reposo mínimo para estimar el ruido de fondo
DIF_PICOS_MAX = 1              # |picos global - picos adaptativo|
CV_IPI_MAX = 0.35              # irregularidad del ritmo entre contracciones
ESPIGAS_MAX_POR_10S = 1.0      # eventos de espiga por cada 10 s

# --- Contracción -> se descarta solo esa contracción
FRAC_ONSET = 0.25              # inicio/fin = 25 % entre la base y el pico
DUR_CONTR_MIN_S, DUR_CONTR_MAX_S = 0.3, 4.0
SNR_CONTR_MIN_DB = 10.0         # contracción vs. piso de ruido
CRESTA_MAX = 8.0               # máx|x| / RMS ; >8 indica espiga o artefacto
AMP_REL_MIN, AMP_REL_MAX = 0.30, 2.5   # pico / mediana de picos del archivo
MDF_MIN, MDF_MAX = 35.0, 200.0         # frecuencia mediana plausible para sEMG
K_ESPIGA = 10.0                # |x| > K · sigma robusta de la actividad = espiga

# --- Ventanas
VENTANA_S = 0.250
PASO_S = 0.125
N_CONTR_BASAL = 3              # primeras contracciones válidas de la 1.ª serie = referencia
INCLUIR_REVISAR = False        # True para usar también archivos REVISAR

# --- Sujeto/pierna
MIN_SERIES_VALIDAS = 3
Z_ATIPICO = 3.0                # |z robusto| para marcar un sujeto como atípico
MDF_CAIDA_MIN = 0.95           # si MDF final/inicial de serie > 0.95: sin evidencia de fatiga


# =====================================================================
# UTILIDADES
# =====================================================================
def rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else np.nan


def racha_max(mask):
    """Longitud de la racha más larga de True."""
    if not mask.any():
        return 0
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def frec_mediana_media(seg, fs):
    if len(seg) < 64:
        return np.nan, np.nan
    f, P = welch(seg, fs=fs, nperseg=min(256, len(seg)))
    m = (f >= 20) & (f <= 450)
    f, P = f[m], P[m]
    if P.sum() <= 0:
        return np.nan, np.nan
    acum = np.cumsum(P)
    return float(f[np.searchsorted(acum, acum[-1] / 2)]), float(np.sum(f * P) / np.sum(P))


def ESTADOS_USABLES():
    return ["OK", "REVISAR"] if INCLUIR_REVISAR else ["OK"]


def z_robusto(s):
    med = s.median()
    mad = 1.4826 * (s - med).abs().median()
    mad = max(mad, 0.05 * abs(med))  # evita z inflados cuando el grupo es muy homogéneo
    return (s - med) / mad if mad > 0 else s * 0


# =====================================================================
# NIVEL ARCHIVO: lectura e integridad
# =====================================================================
def leer_crudo(ruta):
    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    if not {"Relative_Time_s", "Voltage_mV"} <= set(df.columns):
        raise ValueError("faltan columnas Relative_Time_s / Voltage_mV")
    t = pd.to_numeric(df["Relative_Time_s"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df["Voltage_mV"], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(v)
    info = {"frac_nan": float(1 - ok.mean())}
    t, v = t[ok], v[ok]

    dt = np.diff(t)
    med_dt = np.median(dt) if len(dt) else 0
    info["fs_estimada"] = float(1 / med_dt) if med_dt > 0 else np.nan
    info["tiempo_no_monotono"] = int((dt <= 0).sum())
    salto = dt > SALTO_MAX_MUESTRAS / FS
    info["huecos"] = int(salto.sum())
    info["muestras_perdidas"] = int(np.round(dt[salto] * FS).sum() - salto.sum())
    info["hueco_max_ms"] = float(dt.max() * 1000) if len(dt) else 0.0
    info["_idx_huecos"] = np.where(dt > 1.5 / FS)[0] + 1   # muestra posterior a cada hueco
    info["duracion_s"] = len(v) / FS

    rango = v.max() - v.min() if len(v) else 0
    eps = 1e-9 + 1e-6 * rango
    info["frac_saturada"] = float(((v >= v.max() - eps) | (v <= v.min() + eps)).mean()) if len(v) else 1.0
    info["plano_max_s"] = racha_max(np.diff(v) == 0) / FS if len(v) > 1 else 0

    f, P = welch(v - v.mean(), fs=FS, nperseg=min(2048, len(v)))
    banda = lambda a, b: P[(f >= a) & (f < b)].sum()
    info["ratio_red"] = float(banda(58, 62) / max(banda(20, 450), 1e-20))
    info["ratio_baja"] = float(banda(0, 20) / max(banda(0, 500), 1e-20))
    info["rango_crudo"] = float(rango)
    return v, info


# =====================================================================
# NIVEL CONTRACCIÓN: segmentación y métricas
# =====================================================================
def segmentar(env, picos):
    """Inicio/fin de cada contracción: cruce del FRAC_ONSET entre base y pico."""
    base = np.percentile(env, 10)
    segs = []
    for p, (i0, i1) in zip(picos, limites_contracciones(env, picos)):
        thr = base + FRAC_ONSET * (env[p] - base)
        izq = np.where(env[i0:p] <= thr)[0]
        der = np.where(env[p:i1 + 1] <= thr)[0]
        on = i0 + izq[-1] if len(izq) else i0
        off = p + der[0] if len(der) else i1
        segs.append((int(on), int(off)))
    return segs


def caracteristicas_ventana(w, fs, thr_zc):
    d = np.diff(w)
    W = np.abs(np.fft.rfft(w * np.hanning(len(w)), n=256)) ** 2
    f = np.fft.rfftfreq(256, 1 / fs)
    m = (f >= 20) & (f <= 450)
    f, W = f[m], W[m]
    acum = np.cumsum(W)
    return {
        "rms": rms(w),
        "mav": float(np.mean(np.abs(w))),
        "wl": float(np.sum(np.abs(d))),
        "zc": int(np.sum((w[:-1] * w[1:] < 0) & (np.abs(d) >= thr_zc))),
        "ssc": int(np.sum((d[:-1] * d[1:] < 0) &
                          ((np.abs(d[:-1]) >= thr_zc) | (np.abs(d[1:]) >= thr_zc)))),
        "mnf": float(np.sum(f * W) / np.sum(W)) if W.sum() > 0 else np.nan,
        "mdf": float(f[np.searchsorted(acum, acum[-1] / 2)]) if W.sum() > 0 else np.nan,
        # Índice espectral de Dimitrov (FInsm5): aumenta con la fatiga
        "fi_nsm5": float(np.sum(W / f) / np.sum(W * f ** 5)) if W.sum() > 0 else np.nan,
        # Potencia baja (20-45 Hz) / alta (95-450 Hz): aumenta con la fatiga
        "ratio_bajo_alto": float(W[f <= 45].sum() / max(W[f >= 95].sum(), 1e-20)),
    }


def analizar_archivo(ruta, sujeto, pierna, serie):
    v, info = leer_crudo(ruta)
    reg = {"Sujeto": sujeto, "Pierna": pierna, "Serie": serie,
           "Archivo": os.path.basename(ruta),
           **{k: val for k, val in info.items() if not k.startswith("_")}}
    excluir, revisar = [], []

    # ---- Integridad
    if info["frac_nan"] > NAN_MAX:
        excluir.append(f"NaN {info['frac_nan']:.1%}")
    if info["duracion_s"] < DURACION_MIN_S:
        excluir.append(f"corto {info['duracion_s']:.1f}s")
    if info["frac_saturada"] > SATURACION_MAX:
        excluir.append(f"saturación {info['frac_saturada']:.2%}")
    if excluir and info["duracion_s"] < 3:
        reg.update(estado="EXCLUIR", motivos="; ".join(excluir))
        return reg, [], [], None

    if not np.isfinite(info["fs_estimada"]) or abs(info["fs_estimada"] - FS) / FS > TOL_FS:
        revisar.append(f"fs={info['fs_estimada']:.0f}Hz")
    # Huecos pequeños: se descartan solo las ventanas que los contienen.
    # Muchos huecos o huecos largos: el archivo pasa a REVISAR.
    if info["tiempo_no_monotono"]:
        revisar.append(f"tiempo no monótono ({info['tiempo_no_monotono']})")
    if (info["muestras_perdidas"] > PERDIDAS_MAX * len(v)) or info["hueco_max_ms"] > HUECO_MAX_MS:
        revisar.append(f"pérdida de datos ({info['muestras_perdidas']} muestras, "
                       f"máx {info['hueco_max_ms']:.0f} ms)")
    huecos = np.zeros(len(v), bool)
    huecos[info.pop("_idx_huecos")] = True
    reg.pop("_idx_huecos", None)
    if info["plano_max_s"] > PLANO_MAX_S:
        revisar.append(f"tramo plano {info['plano_max_s']:.1f}s")
    if info["ratio_red"] > RATIO_RED_MAX:
        revisar.append(f"ruido 60Hz {info['ratio_red']:.2f}")
    if info["ratio_baja"] > RATIO_BAJA_MAX:
        revisar.append(f"potencia <20Hz {info['ratio_baja']:.2f}")

    # ---- Filtrado + envolvente robusta a espigas (solo para detectar)
    x = acondicionar(v)
    env0 = calcular_envolvente(x)
    activo0 = env0 > np.percentile(env0, 50)
    sigma = 1.4826 * np.median(np.abs(x[activo0]))
    thr_esp = K_ESPIGA * sigma
    idx = np.where(np.abs(x) > thr_esp)[0]
    n_esp = 0 if len(idx) == 0 else 1 + int((np.diff(idx) > 0.05 * FS).sum())
    reg["espigas_por_10s"] = 10 * n_esp / info["duracion_s"]
    if reg["espigas_por_10s"] > ESPIGAS_MAX_POR_10S:
        revisar.append(f"espigas {reg['espigas_por_10s']:.1f}/10s")

    env = calcular_envolvente(np.clip(x, -thr_esp, thr_esp))
    # Techo robusto: un artefacto enorme no debe ocultar las contracciones vecinas
    p0, _ = detectar_adaptativo(env)
    env_det = np.minimum(env, 1.5 * np.median(env[p0])) if len(p0) else env
    picos, _ = detectar_adaptativo(env_det)
    picos_g, _ = detectar_global(env_det)
    reg["picos_adaptativo"], reg["picos_global"] = len(picos), len(picos_g)
    if abs(len(picos) - len(picos_g)) > DIF_PICOS_MAX:
        revisar.append(f"picos g/a {len(picos_g)}/{len(picos)}")
    if len(picos) == 0:
        excluir.append("sin contracciones")
        reg.update(estado="EXCLUIR", motivos="; ".join(excluir + revisar))
        return reg, [], [], (x, env, [], [], [])

    if len(picos) > 2:
        ipi = np.diff(picos) / FS
        reg["ipi_mediana_s"] = float(np.median(ipi))
        reg["cv_ipi"] = float(np.std(ipi) / np.mean(ipi))
        if reg["cv_ipi"] > CV_IPI_MAX:
            revisar.append(f"ritmo irregular CV={reg['cv_ipi']:.2f}")

    segs = segmentar(env_det, picos)

    # ---- Ruido de fondo (reposo)
    activo = np.zeros(len(x), bool)
    for on, off in segs:
        activo[on:off] = True
    base = np.percentile(env, 10)
    reposo = (~activo) & (env < base + 0.15 * (np.median(env[picos]) - base))
    borde = int(0.2 * FS)
    reposo[:borde] = reposo[-borde:] = False
    rms_rep = rms(x[reposo]) if reposo.sum() >= REPOSO_MIN_S * FS else np.nan
    reg["reposo_s"] = reposo.sum() / FS
    reg["rms_reposo"] = rms_rep

    # Piso de ruido eléctrico: RMS de la ventana de 0,5 s más tranquila del registro.
    # (El "reposo" entre contracciones puede tener actividad residual; no es ruido.)
    n05 = int(0.5 * FS)
    x2 = np.square(x[borde:-borde]) if len(x) > 2 * borde + n05 else np.square(x)
    rms_ruido = float(np.sqrt(np.min(np.convolve(x2, np.ones(n05) / n05, mode="valid"))))
    reg["rms_ruido"] = rms_ruido
    thr_zc = rms_ruido

    # ---- Contracciones
    med_pico = np.median(env[picos])
    contr, ventanas, validas = [], [], []
    for k, (p, (on, off)) in enumerate(zip(picos, segs)):
        seg = x[on:off]
        dur = (off - on) / FS
        r = rms(seg)
        cresta = np.max(np.abs(seg)) / r if r > 0 else np.inf
        snr = 20 * np.log10(r / rms_ruido) if rms_ruido > 0 else np.nan
        contraste = 20 * np.log10(r / rms_rep) if np.isfinite(rms_rep) and rms_rep > 0 else np.nan
        amp_rel = env[p] / med_pico
        mdf, mnf = frec_mediana_media(seg, FS)

        mot = []
        if not DUR_CONTR_MIN_S <= dur <= DUR_CONTR_MAX_S:
            mot.append("duración")
        if np.isfinite(snr) and snr < SNR_CONTR_MIN_DB:
            mot.append("SNR")
        if cresta > CRESTA_MAX:
            mot.append("espiga")
        if not AMP_REL_MIN <= amp_rel <= AMP_REL_MAX:
            mot.append("amplitud atípica")
        if not (np.isfinite(mdf) and MDF_MIN <= mdf <= MDF_MAX):
            mot.append("MDF")
        valida = not mot
        validas.append(valida)
        contr.append({"Sujeto": sujeto, "Pierna": pierna, "Serie": serie,
                      "Archivo": reg["Archivo"], "contraccion": k,
                      "t_inicio": on / FS, "t_pico": p / FS, "duracion_s": dur,
                      "rms": r, "snr_db": snr, "contraste_db": contraste, "cresta": cresta, "amp_rel": amp_rel,
                      "mdf": mdf, "mnf": mnf, "valida": valida, "motivos": ", ".join(mot)})

        if not valida:
            continue
        n_w, paso = int(VENTANA_S * FS), int(PASO_S * FS)
        for i in range(on, off - n_w + 1, paso):
            w = x[i:i + n_w]
            rw = rms(w)
            if (rw <= 0 or huecos[i + 1:i + n_w].any() or np.max(np.abs(w)) > thr_esp
                    or np.max(np.abs(w)) / rw > CRESTA_MAX):
                continue
            ventanas.append({"Sujeto": sujeto, "Pierna": pierna, "Serie": serie,
                             "Archivo": reg["Archivo"], "contraccion": k,
                             "t_ventana": i / FS, **caracteristicas_ventana(w, FS, thr_zc)})

    n_val = int(sum(validas))
    reg["contr_detectadas"], reg["contr_validas"] = len(picos), n_val
    reg["snr_mediana_db"] = float(np.nanmedian([c["snr_db"] for c in contr]))
    reg["contraste_mediana_db"] = float(np.nanmedian([c["contraste_db"] for c in contr])) \
        if np.isfinite(rms_rep) else np.nan
    if not np.isfinite(rms_rep):
        revisar.append("sin reposo entre contracciones")
    elif reg["contraste_mediana_db"] < CONTRASTE_MIN_DB:
        revisar.append(f"relajación incompleta (contraste {reg['contraste_mediana_db']:.1f} dB)")
    reg["mdf_mediana"] = float(np.nanmedian([c["mdf"] for c in contr if c["valida"]])) if n_val else np.nan
    reg["dur_contr_mediana_s"] = float(np.median([c["duracion_s"] for c in contr]))
    if np.isfinite(reg["snr_mediana_db"]) and reg["snr_mediana_db"] < SNR_ARCHIVO_MIN_DB:
        revisar.append(f"SNR {reg['snr_mediana_db']:.1f}dB")
    if n_val < MIN_CONTR_VALIDAS:
        excluir.append(f"solo {n_val} contracciones válidas")
    if n_val / len(picos) < FRAC_CONTR_VALIDAS_MIN:
        excluir.append(f"{n_val}/{len(picos)} contracciones válidas")

    reg["estado"] = "EXCLUIR" if excluir else ("REVISAR" if revisar else "OK")
    reg["motivos"] = "; ".join(excluir + revisar)
    return reg, contr, ventanas, (x, env, picos, segs, validas)


# =====================================================================
# FIGURA DE REVISIÓN
# =====================================================================
def figura_revision(reg, datos, ruta_png):
    x, env, picos, segs, validas = datos
    t = np.arange(len(x)) / FS
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{reg['Sujeto']} | {reg['Pierna']} | Serie {reg['Serie']} — {reg['estado']}",
                 fontweight="bold")
    axs[0].set_title(reg["motivos"], fontsize=9, loc="left")
    lim = np.percentile(np.abs(x), 99.9) * 1.2
    axs[0].plot(t, x, color="#444", lw=0.4)
    axs[0].set_ylim(-lim, lim)
    axs[1].plot(t, env, color="#1f5f9e", lw=1.5)
    for (on, off), ok in zip(segs, validas):
        for ax in axs:
            ax.axvspan(on / FS, off / FS, color="#2e7d32" if ok else "#c62828", alpha=0.18, lw=0)
    if len(picos):
        axs[1].plot(t[picos], env[picos], "v", color="k")
    axs[0].set_ylabel("Señal filtrada")
    axs[1].set_ylabel("Envolvente")
    axs[1].set_xlabel("Tiempo (s)  —  verde: válida, rojo: excluida")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    plt.close(fig)


# =====================================================================
# NIVEL SUJETO/PIERNA, NORMALIZACIÓN Y ETIQUETAS
# =====================================================================
def construir_dataset(dfw, dfa, estados=None):
    usables = dfa[dfa.estado.isin(estados or ESTADOS_USABLES())].Archivo
    dfw = dfw[dfw.Archivo.isin(usables)].copy()
    partes = []
    for (s, p), g in dfw.groupby(["Sujeto", "Pierna"]):
        g = g.copy()
        orden = g[["Serie", "contraccion"]].drop_duplicates().sort_values(["Serie", "contraccion"])
        orden["idx_global"] = np.arange(len(orden))
        g = g.merge(orden, on=["Serie", "contraccion"])
        n = max(len(orden) - 1, 1)
        g["pos_global"] = g.idx_global / n

        n_por_serie = orden.groupby("Serie").contraccion.transform("count").values
        rank_serie = orden.groupby("Serie").cumcount().values
        orden["pos_en_serie"] = rank_serie / np.maximum(n_por_serie - 1, 1)
        g = g.merge(orden[["Serie", "contraccion", "pos_en_serie"]], on=["Serie", "contraccion"])

        basal = g[g.idx_global < N_CONTR_BASAL]
        for c in ["rms", "mav", "wl", "mnf", "mdf", "fi_nsm5", "ratio_bajo_alto"]:
            g[c + "_n"] = g[c] / basal[c].median()
        partes.append(g)
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True)
    # Etiqueta principal: posición DENTRO de cada serie.
    # En este dataset la MDF se recupera entre series (descanso), pero cae de forma
    # consistente del inicio al final de cada serie -> la fatiga es intra-serie.
    out["etiqueta"] = np.where(out.pos_en_serie <= 0.25, 0,
                               np.where(out.pos_en_serie >= 0.75, 1, np.nan))
    # Alternativa (no recomendada aquí): extremos de todo el protocolo
    out["etiqueta_global"] = np.where(out.pos_global <= 0.25, 0,
                                      np.where(out.pos_global >= 0.75, 1, np.nan))
    return out


def caida_intra_serie(w, col):
    """Mediana (sobre series) de col en el último cuarto / primer cuarto de cada serie."""
    c = w.groupby(["Serie", "contraccion", "pos_en_serie"])[col].median().reset_index()
    r = []
    for _, d in c.groupby("Serie"):
        a, b = d[d.pos_en_serie <= 0.25][col], d[d.pos_en_serie >= 0.75][col]
        if len(a) and len(b):
            r.append(b.median() / a.median())
    return float(np.median(r)) if r else np.nan


def resumen_sujetos(dfa, dfc, dfw):
    filas = []
    for (s, p), g in dfa.groupby(["Sujeto", "Pierna"]):
        usable = g[g.estado.isin(ESTADOS_USABLES())]
        series = sorted(usable.Serie.astype(str))
        c = dfc[(dfc.Sujeto == s) & (dfc.Pierna == p)]
        w = dfw[(dfw.Sujeto == s) & (dfw.Pierna == p)] if len(dfw) else pd.DataFrame()
        bas = w[w.idx_global < N_CONTR_BASAL] if len(w) else w
        motivos = []
        if len(usable) < MIN_SERIES_VALIDAS:
            motivos.append(f"solo {len(usable)} series usables")
        if "1" not in series:
            motivos.append("falta la serie 1 (sin referencia basal)")
        if usable.Serie.astype(str).duplicated().any():
            motivos.append("series duplicadas")
        if any(sv in ("0", "NA") for sv in series):
            motivos.append("numeración de serie inválida")
        filas.append({
            "Sujeto": s, "Pierna": p, "archivos": len(g),
            "OK": int((g.estado == "OK").sum()), "REVISAR": int((g.estado == "REVISAR").sum()),
            "EXCLUIR": int((g.estado == "EXCLUIR").sum()), "series_usables": ",".join(series),
            "contr_validas": int(c.valida.sum()) if len(c) else 0,
            "ventanas": len(w),
            "ventanas_no_fatiga": int((w.etiqueta == 0).sum()) if len(w) else 0,
            "ventanas_fatiga": int((w.etiqueta == 1).sum()) if len(w) else 0,
            "mdf_final_vs_inicio_serie": caida_intra_serie(w, "mdf") if len(w) else np.nan,
            "rms_final_vs_inicio_serie": caida_intra_serie(w, "rms") if len(w) else np.nan,
            "mdf_basal": bas.mdf.median() if len(bas) else np.nan,
            "rms_basal": bas.rms.median() if len(bas) else np.nan,
            "snr_mediana_db": g.snr_mediana_db.median(),
            "ipi_mediana_s": g.ipi_mediana_s.median() if "ipi_mediana_s" in g else np.nan,
            "dur_contr_mediana_s": g.dur_contr_mediana_s.median(),
            "pendiente_mdf_n_por_serie": (np.polyfit(w.groupby("Serie").mdf_n.median().index.astype(float),
                                                     w.groupby("Serie").mdf_n.median().values, 1)[0]
                                          if len(w) and w.Serie.nunique() >= 2 else np.nan),
            "motivos_pierna": "; ".join(motivos),
        })
    R = pd.DataFrame(filas)
    for col in ["mdf_basal", "snr_mediana_db", "ipi_mediana_s", "dur_contr_mediana_s"]:
        R["z_" + col] = z_robusto(R[col]).round(2)
    atip = lambda r: ", ".join(c[2:] for c in R.columns if c.startswith("z_") and abs(r[c]) > Z_ATIPICO)
    R["atipico_en"] = R.apply(atip, axis=1)
    R["sin_evidencia_fatiga"] = R.mdf_final_vs_inicio_serie > MDF_CAIDA_MIN
    R["apto_entrenamiento"] = (R.motivos_pierna == "") & (R.ventanas_fatiga > 0) & (R.ventanas_no_fatiga > 0)
    return R


def figura_homogeneidad(dfw, dfc, dfa, ruta_png):
    dfw = dfw.copy()
    dfw["grupo"] = dfw.Sujeto.str[:10] + "-" + dfw.Pierna.str[0]
    grupos = sorted(dfw.grupo.unique())
    bas = dfw[dfw.idx_global < N_CONTR_BASAL]
    fig, axs = plt.subplots(4, 2, figsize=(18, 20))

    def caja(ax, df, col, titulo, log=False):
        datos = [df[df.grupo == gname][col].dropna().values for gname in grupos]
        ax.boxplot(datos, showfliers=False)
        ax.set_xticks(range(1, len(grupos) + 1))
        ax.set_xticklabels(grupos, rotation=75, fontsize=7)
        ax.set_title(titulo, fontsize=10)
        if log:
            ax.set_yscale("log")
        ax.grid(True, ls="--", alpha=0.4)

    caja(axs[0, 0], bas, "mdf", "MDF basal (Hz) — debería ser similar entre sujetos")
    caja(axs[0, 1], bas, "rms", "RMS basal (unidades crudas, log) — justifica normalizar", log=True)

    c = dfc.copy()
    c["grupo"] = c.Sujeto.str[:10] + "-" + c.Pierna.str[0]
    caja(axs[1, 0], c, "snr_db", "SNR por contracción (dB)")
    axs[1, 0].axhline(SNR_ARCHIVO_MIN_DB, color="r", ls="--")
    caja(axs[1, 1], c, "duracion_s", "Duración de contracción (s) — ritmo/protocolo")

    dfw["tramo"] = np.minimum((dfw.pos_en_serie * 5).astype(int), 4) + 1
    for col, ax, tit in [("mdf_n", axs[2, 0], "MDF normalizada a lo largo de la serie"),
                         ("rms_n", axs[2, 1], "RMS normalizada a lo largo de la serie")]:
        prog = dfw.groupby(["grupo", "tramo"])[col].median().reset_index()
        for gname, gg in prog.groupby("grupo"):
            ax.plot(gg.tramo, gg[col], color="gray", alpha=0.4, lw=1)
        med = prog.groupby("tramo")[col].median()
        ax.plot(med.index, med.values, "o-", color="#1f5f9e", lw=3, label="mediana del grupo")
        ax.axhline(1, color="k", ls=":")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
        ax.set_xlabel("Posición dentro de la serie")
        ax.set_title(tit + " (cada línea = sujeto-pierna)", fontsize=10)
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)

    for col, ax, tit in [("mdf_n", axs[3, 0], "MDF normalizada por serie"),
                         ("rms_n", axs[3, 1], "RMS normalizada por serie")]:
        prog = dfw.groupby(["grupo", "Serie"])[col].median().reset_index()
        for gname, gg in prog.groupby("grupo"):
            ax.plot(gg.Serie.astype(float), gg[col], color="gray", alpha=0.4, lw=1)
        med = prog.groupby("Serie")[col].median()
        ax.plot(med.index.astype(float), med.values, "o-", color="#c62828", lw=3, label="mediana del grupo")
        ax.axhline(1, color="k", ls=":")
        ax.set_xlabel("Serie")
        ax.set_title(tit + " (cada línea = sujeto-pierna)", fontsize=10)
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=110)
    plt.close(fig)


# =====================================================================
# FLUJO PRINCIPAL
# =====================================================================
def main():
    os.makedirs(os.path.join(RUTA_QC, "revisar"), exist_ok=True)
    if not os.path.isdir(RUTA_DATASET):
        print(f"No se encontró {RUTA_DATASET}")
        return

    archivos_qc, contracciones, ventanas = [], [], []
    for carpeta in sorted(f.path for f in os.scandir(RUTA_DATASET) if f.is_dir()):
        sujeto = os.path.basename(carpeta)
        for ruta in sorted(glob.glob(os.path.join(carpeta, "*.csv"))):
            nombre = os.path.splitext(os.path.basename(ruta))[0].lower()
            pierna, serie = extraer_pierna(nombre), extraer_repeticion(nombre)
            try:
                reg, c, w, datos = analizar_archivo(ruta, sujeto, pierna, serie)
            except Exception as e:
                reg, c, w, datos = ({"Sujeto": sujeto, "Pierna": pierna, "Serie": serie,
                                     "Archivo": os.path.basename(ruta), "estado": "EXCLUIR",
                                     "motivos": f"error: {e}"}, [], [], None)
            archivos_qc.append(reg)
            contracciones += c
            ventanas += w
            print(f"{reg['estado']:8s} {reg['Archivo']:35s} {reg.get('motivos', '')}")
            if reg["estado"] != "OK" and datos is not None:
                figura_revision(reg, datos, os.path.join(
                    RUTA_QC, "revisar", f"{limpiar(sujeto)}_{pierna}_S{serie}_{reg['estado']}.png"))

    dfa = pd.DataFrame(archivos_qc)
    # series duplicadas dentro de sujeto-pierna -> REVISAR
    dup = dfa.duplicated(["Sujeto", "Pierna", "Serie"], keep=False) & (dfa.estado == "OK")
    dfa.loc[dup, "estado"] = "REVISAR"
    dfa.loc[dup, "motivos"] = dfa.loc[dup, "motivos"].fillna("") + "; serie duplicada"

    dfc = pd.DataFrame(contracciones)
    dfw = construir_dataset(pd.DataFrame(ventanas), dfa)
    dfs = resumen_sujetos(dfa, dfc, dfw)

    # Solo quedan en el dataset final los sujeto-pierna aptos
    aptos = dfs[dfs.apto_entrenamiento][["Sujeto", "Pierna"]]
    dfw_final = dfw.merge(aptos, on=["Sujeto", "Pierna"]) if len(dfw) else dfw

    dfa.to_csv(os.path.join(RUTA_QC, "qc_archivos.csv"), index=False, encoding="utf-8-sig")
    dfc.to_csv(os.path.join(RUTA_QC, "qc_contracciones.csv"), index=False, encoding="utf-8-sig")
    dfs.to_csv(os.path.join(RUTA_QC, "qc_sujetos.csv"), index=False, encoding="utf-8-sig")
    dfw_final.to_csv(os.path.join(RUTA_QC, "dataset_ventanas.csv"), index=False, encoding="utf-8-sig")

    if len(dfw):
        figura_homogeneidad(dfw, dfc, dfa, os.path.join(RUTA_QC, "homogeneidad.png"))

    # Dataset COMPLETO (todas las ventanas válidas, sin filtrar archivo ni sujeto-pierna)
    # con banderas, para comparar escenarios de exclusión en el entrenamiento.
    dfw_todo = construir_dataset(pd.DataFrame(ventanas), dfa, estados=["OK", "REVISAR", "EXCLUIR"])
    if len(dfw_todo):
        dfw_todo = dfw_todo.merge(dfa[["Archivo", "Sujeto", "estado", "motivos"]]
                                  .rename(columns={"estado": "estado_archivo", "motivos": "motivos_archivo"}),
                                  on=["Archivo", "Sujeto"], how="left")
        dfw_todo = dfw_todo.merge(dfs[["Sujeto", "Pierna", "apto_entrenamiento",
                                       "sin_evidencia_fatiga", "atipico_en"]],
                                  on=["Sujeto", "Pierna"], how="left")
        dfw_todo.to_csv(os.path.join(RUTA_QC, "dataset_ventanas_completo.csv"),
                        index=False, encoding="utf-8-sig")

    print("\n=== RESUMEN ===")
    print(dfa.estado.value_counts().to_string())
    if len(dfc):
        print(f"Contracciones válidas: {dfc.valida.sum()}/{len(dfc)}")
    print(f"Sujeto-pierna aptos: {int(dfs.apto_entrenamiento.sum())}/{len(dfs)}")
    print(f"Ventanas en el dataset final: {len(dfw_final)}")
    atip = dfs[dfs.atipico_en != ""]
    if len(atip):
        print("Atípicos respecto al grupo:\n" + atip[["Sujeto", "Pierna", "atipico_en"]].to_string(index=False))


if __name__ == "__main__":
    main()