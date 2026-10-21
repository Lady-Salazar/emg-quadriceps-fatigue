

# -*- coding: utf-8 -*-
"""
Massive processing of surface EMG signals for fatigue analysis.

Expected structure:
    EMG_RAW/
        Subject1/  *.csv  (columns: Relative_Time_s, Voltage_mV ; fs = 1000 Hz)
        Subject2/  *.csv
        ...

For each CSV:
    1. Removes offset, applies 50 Hz and 60 Hz Notch filters, and a 20 Hz high-pass filter.
    2. Rectifies and applies a 1 Hz Butterworth low-pass filter -> envelope.
    3. Detects peaks with a GLOBAL threshold and an ADAPTIVE threshold.
    4. Saves a figure with 3 subplots in ./Resultados.
Finally, saves ./Resultados/resumen_picos.csv
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: más rápido y estable en lotes
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, find_peaks
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d

# ==========================================
# CONFIGURACIÓN
# ==========================================
# CORREGIDO: Se usa ruta relativa para que funcione en cualquier PC o en Colab.
# Si se define la variable de entorno EMG_RAW, se usará esa ruta.
RUTA_DATASET = os.environ.get("EMG_RAW", "./EMG_RAW")
RUTA_RESULTADOS = "./Resultados"

FS = 1000.0                 # Hz
FRECS_NOTCH = (50.0, 60.0)  # 50 Hz (indicación del curso) y 60 Hz (red eléctrica de Ecuador)
Q_NOTCH = 30.0              # factor de calidad del notch (ancho ~2 Hz)
FC_PASA_ALTOS = 20.0        # Hz
FC_ENVOLVENTE = 1.0         # Hz (indicación del curso)
ORDEN_BUTTER = 4

SEPARACION_MIN_S = 1.5      # separación mínima entre contracciones (s)
UMBRAL_GLOBAL = 0.30        # fracción del rango (percentil 10 -> máximo)
PROMINENCIA_MIN = 0.10      # fracción del rango de la envolvente
VENTANA_ADAPT_S = 6.0       # ventana del umbral adaptativo (s)
UMBRAL_ADAPT = 0.50         # fracción del rango local (mín local -> máx local)
RANGO_LOCAL_MIN = 0.08      # ignora zonas sin actividad (fracción del rango global)


# ==========================================
# LECTURA
# ==========================================
def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    faltan = {"Relative_Time_s", "Voltage_mV"} - set(df.columns)
    if faltan:
        raise ValueError(f"faltan columnas {faltan}")

    df = df[["Relative_Time_s", "Voltage_mV"]].apply(pd.to_numeric, errors="coerce").dropna()
    t = df["Relative_Time_s"].to_numpy(dtype=float)
    v = df["Voltage_mV"].to_numpy(dtype=float)

    if len(v) < 3 * FS:
        raise ValueError(f"señal demasiado corta ({len(v)} muestras)")

    # Verifica que el eje temporal sea coherente con fs = 1000 Hz
    dt = np.median(np.diff(t))
    if dt <= 0 or abs(1.0 / dt - FS) / FS > 0.05:
        print(f"   Aviso: la columna de tiempo no coincide con {FS:.0f} Hz; se reconstruye.")
        t = np.arange(len(v)) / FS
    return t, v


# ==========================================
# FILTRADO Y ENVOLVENTE
# ==========================================
def acondicionar(v, fs=FS):
    """Quita offset, aplica Notch(es) y pasa-altos. Fase cero (filtfilt)."""
    x = v - np.mean(v)
    for f0 in FRECS_NOTCH:
        if f0 < fs / 2:
            b, a = iirnotch(f0, Q_NOTCH, fs=fs)
            x = filtfilt(b, a, x)
    b, a = butter(ORDEN_BUTTER, FC_PASA_ALTOS, btype="highpass", fs=fs)
    return filtfilt(b, a, x)


def calcular_envolvente(x, fs=FS):
    """Rectificación de onda completa + pasa-bajos Butterworth."""
    b, a = butter(ORDEN_BUTTER, FC_ENVOLVENTE, btype="lowpass", fs=fs)
    env = filtfilt(b, a, np.abs(x))
    return np.clip(env, 0, None)  # filtfilt puede dar valores levemente negativos


# ==========================================
# DETECCIÓN DE PICOS
# ==========================================
def detectar_global(env, fs=FS):
    """Un único umbral para toda la señal."""
    base = np.percentile(env, 10)
    rango = env.max() - base
    umbral = base + UMBRAL_GLOBAL * rango
    picos, _ = find_peaks(env,
                          height=umbral,
                          distance=max(1, int(SEPARACION_MIN_S * fs)),
                          prominence=PROMINENCIA_MIN * rango)
    return picos, umbral


def detectar_adaptativo(env, fs=FS):
    """Umbral que sigue el mínimo/máximo local (útil si la amplitud cambia con la fatiga)."""
    n = max(3, int(VENTANA_ADAPT_S * fs))
    lo = uniform_filter1d(minimum_filter1d(env, n, mode="nearest"), n, mode="nearest")
    hi = uniform_filter1d(maximum_filter1d(env, n, mode="nearest"), n, mode="nearest")
    umbral_local = lo + UMBRAL_ADAPT * (hi - lo)

    rango_global = env.max() - env.min()
    candidatos, _ = find_peaks(env,
                               distance=max(1, int(SEPARACION_MIN_S * fs)),
                               prominence=0.5 * PROMINENCIA_MIN * rango_global)
    validos = (env[candidatos] > umbral_local[candidatos]) & \
              ((hi - lo)[candidatos] > RANGO_LOCAL_MIN * rango_global)
    return candidatos[validos], umbral_local


def limites_contracciones(env, picos):
    """Devuelve (inicio, fin) de cada contracción: de valle a valle."""
    if len(picos) == 0:
        return []
    valles = [p0 + int(np.argmin(env[p0:p1])) for p0, p1 in zip(picos[:-1], picos[1:])]
    sep = np.median(np.diff(picos)) if len(picos) > 1 else len(env) / 4
    inicio = max(0, int(picos[0] - sep / 2))
    fin = min(len(env) - 1, int(picos[-1] + sep / 2))
    bordes = [inicio] + valles + [fin]
    return list(zip(bordes[:-1], bordes[1:]))


# ==========================================
# ETIQUETADO DESDE EL NOMBRE DEL ARCHIVO
# ==========================================
IZQ_CODIGOS = {"i", "iz", "l", "lt", "left", "izq", "izquierda", "izquierdo"}
DER_CODIGOS = {"d", "dr", "de", "r", "rt", "right", "der", "derecha", "derecho"}


def extraer_pierna(nombre):
    if "izq" in nombre or "left" in nombre:
        return "Izquierda"
    if "der" in nombre or "right" in nombre:
        return "Derecha"
    # Códigos cortos aislados (p.ej. "P01_set03_dr", "S2_i_4")
    for tok in reversed(re.findall(r"[a-z]+", nombre)):
        if tok in IZQ_CODIGOS:
            return "Izquierda"
        if tok in DER_CODIGOS:
            return "Derecha"
    return "Desconocida"


def extraer_repeticion(nombre):
    m = re.search(r"(?:repeticion|repetición|rep|set|serie|trial|intento|r)[_\-\s]*(\d+)", nombre)
    if m:
        return str(int(m.group(1)))
    numeros = re.findall(r"\d+", nombre)
    return str(int(numeros[-1])) if numeros else "NA"


def limpiar(texto):
    return re.sub(r"[^\w\-]+", "_", str(texto)).strip("_")


# ==========================================
# GRÁFICAS
# ==========================================
ESTILO_GLOBAL = {
    "bandas": ("#f6c9b8", "#fbe6dc"),
    "linea": "#e07a5f",
    "marcador": "#c62828",
    "umbral": "#7a7a7a",
    "relleno": False,
    "leyenda": "Normal: umbral global, separación ≥ " + f"{SEPARACION_MIN_S:g}" + " s → {n} reps",
}
ESTILO_ADAPT = {
    "bandas": ("#bcd7f0", "#e1eefa"),
    "linea": "#1f5f9e",
    "marcador": "#0d3b66",
    "umbral": "#f28e2b",
    "relleno": True,
    "leyenda": "Adaptativo: umbral local (ventana " + f"{VENTANA_ADAPT_S:g}" + " s) → {n} reps",
}


def _panel_envolvente(ax, t, env, picos, umbral, estilo):
    for k, (i0, i1) in enumerate(limites_contracciones(env, picos)):
        ax.axvspan(t[i0], t[i1], color=estilo["bandas"][k % 2], alpha=0.45, lw=0)

    if estilo["relleno"]:
        ax.fill_between(t, env, color=estilo["linea"], alpha=0.15, lw=0)
    ax.plot(t, env, color=estilo["linea"], lw=1.8)

    if np.ndim(umbral) == 0:
        ax.axhline(umbral, color=estilo["umbral"], ls="--", lw=1, label="Umbral global")
    else:
        ax.plot(t, umbral, color=estilo["umbral"], ls="--", lw=1, label="Umbral adaptativo")

    ax.plot(t[picos], env[picos] + 3, "v", color=estilo["marcador"], ms=8)
    for k, p in enumerate(picos, start=1):
        ax.text(t[p], env[p] + 7, str(k), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=estilo["marcador"])

    ax.set_ylim(0, 118)
    ax.set_ylabel("Envolvente (% máx)")
    ax.text(0.005, 0.97, estilo["leyenda"].format(n=len(picos)), transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax.grid(True, ls="--", alpha=0.4)


def graficar(t, x, env_pct, pg, ug, pa, ua, sujeto, pierna, rep, nombre_csv, ruta_png):
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                            gridspec_kw={"height_ratios": [1.3, 1, 1]})
    fig.suptitle(f"Sujeto: {sujeto} | Pierna: {pierna} | Repetición: {rep}",
                 fontsize=15, fontweight="bold")

    # --- 1) Señal filtrada (Notch + pasa-altos) ---
    ax = axs[0]
    ax.plot(t, x, color="#4d4d4d", lw=0.4)
    lim = np.percentile(np.abs(x), 99.9) * 1.15
    max_abs = np.max(np.abs(x))
    if lim > 0 and max_abs > lim:  # recorta artefactos aislados para que no aplasten la señal
        ax.set_ylim(-lim, lim)
        ax.text(0.995, 1.01, f"máx |x| = {max_abs:.3g} mV (fuera del eje)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#666")
    ax.set_ylabel("Amplitud (mV)")
    ax.set_title(f"{nombre_csv}   |   picos global = {len(pg)}   |   picos adaptativo = {len(pa)}",
                 loc="left", fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)

    # --- 2) Envolvente – detección global ---
    _panel_envolvente(axs[1], t, env_pct, pg, ug, ESTILO_GLOBAL)

    # --- 3) Envolvente – detección adaptativa ---
    _panel_envolvente(axs[2], t, env_pct, pa, ua, ESTILO_ADAPT)
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].set_xlim(t[0], t[-1])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# ==========================================
# PROCESAMIENTO DE UN ARCHIVO
# ==========================================
def procesar_archivo(ruta_csv, sujeto, pierna, rep, ruta_png):
    t, v = leer_csv(ruta_csv)
    x = acondicionar(v)
    env = calcular_envolvente(x)
    env_pct = 100.0 * env / env.max() if env.max() > 0 else env

    pg, ug = detectar_global(env_pct)
    pa, ua = detectar_adaptativo(env_pct)

    graficar(t, x, env_pct, pg, ug, pa, ua, sujeto, pierna, rep,
             os.path.basename(ruta_csv), ruta_png)

    amp_mv = env[pg]  # amplitudes reales (mV) para comparar entre archivos
    pendiente = np.polyfit(np.arange(len(amp_mv)), amp_mv, 1)[0] if len(amp_mv) >= 2 else np.nan
    return {
        "Sujeto": sujeto,
        "Pierna": pierna,
        "Repeticion": rep,
        "Archivo": os.path.basename(ruta_csv),
        "Duracion_s": round(t[-1] - t[0], 2),
        "Picos_global": len(pg),
        "Picos_adaptativo": len(pa),
        "Amp_media_picos_mV": round(float(np.mean(amp_mv)), 5) if len(amp_mv) else np.nan,
        "Pendiente_amp_mV_por_contraccion": round(float(pendiente), 6) if np.isfinite(pendiente) else np.nan,
        "Amp_picos_mV": ";".join(f"{a:.5f}" for a in amp_mv),
        "Tiempos_picos_s": ";".join(f"{t[p]:.2f}" for p in pg),
    }


# ==========================================
# FLUJO PRINCIPAL
# ==========================================
def main():
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)

    if not os.path.isdir(RUTA_DATASET):
        print(f"¡Error! No se encontró la carpeta: {RUTA_DATASET}")
        return

    carpetas = sorted(f.path for f in os.scandir(RUTA_DATASET) if f.is_dir())
    resumen, nombres_usados = [], set()
    n_ok = n_err = 0

    for carpeta in carpetas:
        sujeto = os.path.basename(carpeta)
        archivos = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
        print(f"\n== {sujeto}: {len(archivos)} archivos ==")

        for archivo in archivos:
            nombre = os.path.splitext(os.path.basename(archivo))[0].lower()
            pierna = extraer_pierna(nombre)
            rep = extraer_repeticion(nombre)

            base_png = f"{limpiar(sujeto)}_{pierna}_Rep{rep}"
            nombre_png, k = base_png, 2
            while nombre_png in nombres_usados:  # evita sobrescribir
                nombre_png = f"{base_png}_{k}"
                k += 1
            nombres_usados.add(nombre_png)
            ruta_png = os.path.join(RUTA_RESULTADOS, nombre_png + ".png")

            try:
                fila = procesar_archivo(archivo, sujeto, pierna, rep, ruta_png)
                resumen.append(fila)
                n_ok += 1
                print(f"   OK  {nombre_png}.png  (global={fila['Picos_global']}, "
                      f"adaptativo={fila['Picos_adaptativo']})")
            except Exception as e:
                n_err += 1
                print(f"   ERROR en {os.path.basename(archivo)}: {e}")

    if resumen:
        df = pd.DataFrame(resumen).sort_values(["Sujeto", "Pierna", "Repeticion"])
        df.to_csv(os.path.join(RUTA_RESULTADOS, "resumen_picos.csv"), index=False, encoding="utf-8-sig")

    print(f"\n¡Procesamiento completado! {n_ok} archivos OK, {n_err} con error. "
          f"Revisa la carpeta '{RUTA_RESULTADOS}'.")


if __name__ == "__main__":
    main()