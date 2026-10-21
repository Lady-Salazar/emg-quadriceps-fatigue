# -*- coding: utf-8 -*-
"""
Generates all report figures (1-6 and A1-A3) in rep/figs/.
Requires running, in order: qc_emg.py, train_models.py, and report_analysis.py.
"""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score

# Importamos el script de entrenamiento con su nuevo nombre en inglés
import qc_emg, train_models as em

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
OUT = os.path.join('rep', 'figs'); os.makedirs(OUT, exist_ok=True)
FS = 1000

print("Cargando datos y predicciones...")
d = em.cargar_datos()
D1 = em.escenarios(d)['E1']
DD = d[(d.estado_archivo == 'OK') & (d.apto_entrenamiento == True) & d[em.CARACT].notna().all(axis=1)].copy()

# BÚSQUEDA AUTOMÁTICA DEL ARCHIVO DE PREDICCIONES (para evitar errores con tildes)
archivos = glob.glob('Modelos/predicciones/E1_Reg*.csv')
if not archivos:
    raise FileNotFoundError("No se encontró el archivo de predicciones de Regresión Logística.")
pred = pd.read_csv(archivos[0])

pc = pred.groupby(['Sujeto', 'contr_id']).agg(y=('y', 'first'), s=('s', 'mean')).reset_index()
CODIGO = {s: f'S{i+1:02d}' for i, s in enumerate(sorted(d.Sujeto.unique()))}

# ---------------- Figura 1: DINÁMICA (lee el Excel de QC)
print("Figura 1...")
df_qc = pd.read_csv('QC/qc_archivos.csv')
archivo_ok = df_qc[df_qc.estado == 'OK'].iloc[0]
archivo_rechazado = df_qc[df_qc.estado == 'EXCLUIR'].iloc[0] if any(df_qc.estado == 'EXCLUIR') else df_qc[df_qc.estado == 'REVISAR'].iloc[0]

ruta_ok = os.path.join(em.RUTA_DATASET, archivo_ok['Sujeto'], archivo_ok['Archivo'])
ruta_rechazado = os.path.join(em.RUTA_DATASET, archivo_rechazado['Sujeto'], archivo_rechazado['Archivo'])

def motivo_en_ingles(reg):
    if reg['estado'] == 'OK': return 'all quality criteria passed'
    p = []
    if reg.get('contr_validas') is not None: p.append(f"{reg['contr_validas']}/{reg['contr_detectadas']} valid contractions")
    if reg.get('espigas_por_10s', 0) > 1: p.append(f"impulsive spikes ({reg['espigas_por_10s']:.1f} per 10 s)")
    if reg.get('cv_ipi', 0) > 0.35: p.append(f"irregular rhythm (CV = {reg['cv_ipi']:.2f})")
    return '; '.join(p)

def panel(axs, ruta_csv, sujeto, pierna, serie, titulo):
    reg, _, _, (x, env, picos, segs, val) = qc_emg.analizar_archivo(ruta_csv, sujeto, pierna, serie)
    t = np.arange(len(x)) / FS
    lim = np.percentile(np.abs(x), 99.9) * 1.2
    axs[0].plot(t, x, color='#444', lw=0.35); axs[0].set_ylim(-lim, lim)
    axs[1].plot(t, env, color='#1f5f9e', lw=1.3)
    for (on, off), ok in zip(segs, val):
        for ax in axs: ax.axvspan(on / FS, off / FS, color='#2e7d32' if ok else '#c62828', alpha=0.2, lw=0)
    axs[0].set_title(f"{titulo}\n{motivo_en_ingles(reg)}", fontsize=9, loc='left')
    axs[0].set_ylabel('Filtered sEMG (mV at ADC)'); axs[1].set_ylabel('Envelope'); axs[1].set_xlabel('Time (s)')

fig, axs = plt.subplots(2, 2, figsize=(13, 5.5), sharex='col')
panel(axs[:, 0], ruta_ok, archivo_ok['Sujeto'], archivo_ok['Pierna'], archivo_ok['Serie'], '(a) Accepted trial')
panel(axs[:, 1], ruta_rechazado, archivo_rechazado['Sujeto'], archivo_rechazado['Pierna'], archivo_rechazado['Serie'], f"(b) Rejected trial ({archivo_rechazado['estado']})")

fig.text(0.5, 0.005, 'Green: contraction retained; red: contraction discarded by contraction-level QC', ha='center', fontsize=9)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(f'{OUT}/fig1.png', dpi=170); plt.close()

# ---------------- Figura 2: trayectorias con TODAS las contracciones
print("Figura 2...")
dd = DD.copy()
dd['bin'] = np.minimum((dd.pos_en_serie * 10).astype(int), 9)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
xs = (np.arange(10) + 0.5) * 10
for ax, (c, nm) in zip(axs, [('mdf_n', 'MDF'), ('mnf_n', 'MNF'), ('rms_n', 'RMS'), ('fi_nsm5_n', 'FInsm5')]):
    g = dd.groupby(['Sujeto', 'Pierna', 'bin'])[c].median().apply(np.exp).reset_index()
    pv = g.pivot_table(index=['Sujeto', 'Pierna'], columns='bin', values=c).reindex(columns=range(10))
    m, s = pv.mean(), pv.std()
    ax.fill_between(xs, m - s, m + s, color='#1f5f9e', alpha=0.2); ax.plot(xs, m, 'o-', color='#1f5f9e', lw=2)
    ax.axhline(1, color='k', ls=':', lw=0.8); ax.axvspan(0, 25, color='#2e7d32', alpha=0.07); ax.axvspan(75, 100, color='#c62828', alpha=0.07)
    ax.set_title(nm); ax.set_xlabel('Position within set (%)'); ax.set_xlim(0, 100)
axs[0].set_ylabel('Value relative to baseline')
plt.tight_layout(); plt.savefig(f'{OUT}/fig2.png', dpi=170); plt.close()

# ---------------- Figura 3: esquema LOSO
print("Figura 3...")
n = len(CODIGO)
fig, ax = plt.subplots(figsize=(11, 4.6))
for i in range(n):
    for j in range(n):
        ax.add_patch(plt.Rectangle((j, n - 1 - i), 0.92, 0.8, color='#c62828' if i == j else '#9ec5e8'))
ax.set_xlim(-0.2, n); ax.set_ylim(-0.4, n)
ax.set_yticks([n - 1 - i + 0.4 for i in range(n)]); ax.set_yticklabels([f'Fold {i+1}' for i in range(n)], fontsize=7)
ax.set_xticks([j + 0.46 for j in range(n)]); ax.set_xticklabels([f'S{j+1:02d}' for j in range(n)], fontsize=7)
ax.set_xlabel('Participant (both legs kept together)')
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
ax.legend(handles=[Patch(color='#9ec5e8', label='Training (14 participants)'), Patch(color='#c62828', label='Test (1 held-out participant)')], loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3.png', dpi=170); plt.close()

# ---------------- Figura 4: matriz de confusión
print("Figura 4...")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, P, tit in [(axs[0], pred, '(a) Windows (250 ms)'), (axs[1], pc, '(b) Contractions')]:
    cm = confusion_matrix(P.y, (P.s >= 0.5).astype(int), labels=[0, 1]); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({100*cmn[i,j]:.1f}%)', ha='center', va='center', color='white' if cmn[i, j] > .5 else 'black', fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No fatigue', 'Fatigue']); ax.set_yticks([0, 1]); ax.set_yticklabels(['No fatigue', 'Fatigue'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(tit)
    for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4.png', dpi=170); plt.close()

# ---------------- Figura 5: macro-F1 por sujeto
print("Figura 5...")
ps = []
for suj, g in pred.groupby('Sujeto'):
    gc = pc[pc.Sujeto == suj]
    ps.append(dict(cod=CODIGO[suj], f1_w=f1_score(g.y, g.s >= .5, average='macro'), f1_c=f1_score(gc.y, gc.s >= .5, average='macro')))
PS = pd.DataFrame(ps).sort_values('f1_c')
fig, ax = plt.subplots(figsize=(11, 4)); xx = np.arange(len(PS))
ax.bar(xx - 0.2, PS.f1_w * 100, 0.4, label='Window', color='#9ec5e8'); ax.bar(xx + 0.2, PS.f1_c * 100, 0.4, label='Contraction', color='#1f5f9e')
ax.axhline(PS.f1_w.mean() * 100, color='#9ec5e8', ls='--'); ax.axhline(PS.f1_c.mean() * 100, color='#1f5f9e', ls='--')
ax.set_xticks(xx); ax.set_xticklabels(PS.cod); ax.set_ylim(30, 100); ax.set_ylabel('Macro-F1 (%)'); ax.set_xlabel('Held-out participant'); ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/fig5.png', dpi=170); plt.close()

# ---------------- Figura 6: ablación
print("Figura 6...")
if os.path.exists('rep/resultados_reporte.json'):
    R_abl = json.load(open('rep/resultados_reporte.json'))['ablation']
    names = list(R_abl); w = [R_abl[k]['win'][0] * 100 for k in names]; c = [R_abl[k]['contr'][0] * 100 for k in names]
    fig, ax = plt.subplots(figsize=(10, 4)); yy = np.arange(len(names))[::-1]
    ax.barh(yy + 0.2, w, 0.4, color='#9ec5e8', label='Window'); ax.barh(yy - 0.2, c, 0.4, color='#1f5f9e', label='Contraction')
    for yv, a, b in zip(yy, w, c):
        ax.text(a + 0.3, yv + 0.2, f'{a:.1f}', va='center', fontsize=8); ax.text(b + 0.3, yv - 0.2, f'{b:.1f}', va='center', fontsize=8)
    ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(60, 90); ax.set_xlabel('LOSO accuracy (%)'); ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig6.png', dpi=170); plt.close()
else:
    print("   Aviso: no se encontró resultados_reporte.json, saltando Figura 6.")

# ---------------- Figura A1: comparación de modelos
print("Figura A1...")
C = pd.read_csv('Modelos/resultados_completos.csv')
mp = {'Regresión logística': 'Logistic regression', 'LDA': 'LDA', 'Naive Bayes': 'Naive Bayes', 'KNN (k=25)': 'KNN', 'Random Forest': 'Random forest', 'Gradient Boosting': 'Gradient boosting', 'SVM lineal': 'Linear SVM', 'SVM RBF': 'RBF SVM', 'MLP (64-32)': 'MLP', 'CNN 1D (señal cruda)': '1D CNN'}
C['M'] = C.Modelo.map(mp)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
order = C[C.Escenario == 'E1'].sort_values('contr_acc').M
for ax, col, tit in [(axs[0], 'vent_acc', 'Window accuracy (%)'), (axs[1], 'contr_acc', 'Contraction accuracy (%)')]:
    pv = C.pivot(index='M', columns='Escenario', values=col).loc[order] * 100
    pv = pv[['E0', 'E1', 'E2']]; pv.columns = ['E0', 'E1', 'E2']
    pv.plot.barh(ax=ax, width=0.8, color=['#c9d6e3', '#1f5f9e', '#7fb3e0'])
    ax.set_xlim(60, 90); ax.set_xlabel(tit); ax.set_ylabel(''); ax.grid(axis='x', ls='--', alpha=.4)
axs[1].legend().remove(); axs[0].legend(fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig(f'{OUT}/figA1.png', dpi=170); plt.close()

# ---------------- Figura A2: probabilidad por posición
print("Figura A2...")
D = DD.copy(); lab = D.etiqueta.notna(); mod = em.modelos()['Regresión logística']; D['p'] = np.nan
for s in D.Sujeto.unique():
    tr = lab & (D.Sujeto != s); te = D.Sujeto == s
    m = clone(mod).fit(D.loc[tr, em.CARACT].values, D.loc[tr, 'etiqueta'].astype(int))
    D.loc[te, 'p'] = m.predict_proba(D.loc[te, em.CARACT].values)[:, 1]
cc = D.groupby(['Sujeto', 'contr_id']).agg(pos=('pos_en_serie', 'first'), p=('p', 'mean')).reset_index()
etq = ['0-25%', '25-50%', '50-75%', '75-100%']
cond = [cc.pos <= 0.25, (cc.pos > 0.25) & (cc.pos <= 0.5), (cc.pos > 0.5) & (cc.pos < 0.75), cc.pos >= 0.75]
cc['t'] = np.select(cond, etq, default='')
data = [cc[cc.t == k].p.values for k in etq]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.boxplot(data, showfliers=False); ax.set_xticks(range(1, 5)); ax.set_xticklabels(etq, fontsize=8)
ax.axhline(0.5, color='gray', ls=':'); ax.plot(range(1, 5), [x.mean() for x in data], 'o-', color='#c62828', label='mean')
ax.set_ylabel('Predicted P(fatigue)'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/figA2.png', dpi=170); plt.close()

# ---------------- Figura A3: riesgo–cobertura
print("Figura A3...")
halfw = np.linspace(0, 0.25, 26); cov, acc = [], []
for h in halfw:
    k = ~((pc.s > 0.5 - h) & (pc.s < 0.5 + h)) if h > 0 else np.ones(len(pc), bool)
    cov.append(k.mean() * 100); acc.append(((pc.s[k] >= .5) == pc.y[k]).mean() * 100)
fig, ax = plt.subplots(figsize=(7, 3.8)); ax.plot(cov, acc, 'o-', color='#1f5f9e', ms=3)
ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Accuracy on accepted (%)'); ax.invert_xaxis(); ax.grid(ls='--', alpha=.4)
plt.tight_layout(); plt.savefig(f'{OUT}/figA3.png', dpi=170); plt.close()

print(f"\n[OK] Todas las figuras están en '{OUT}/'.")