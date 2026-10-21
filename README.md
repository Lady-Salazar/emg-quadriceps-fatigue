# EMG-based detection of quadriceps fatigue — Group 01

Surface EMG pipeline for the detection and classification of acute peripheral
fatigue in the quadriceps during repeated unilateral knee extensions. The
repository implements the full analysis reported in the manuscript: signal
processing, four-level quality control, feature extraction, within-set
labelling, leave-one-subject-out (LOSO) classification, and every figure
in the report.

## Repository layout

Proyecto_EMG/
│
├── process_emg.py
│   ├── What it does: Signal processing (50 and 60 Hz notch, 20 Hz high-pass), 1 Hz envelope and
│   │   contraction detection (global and adaptive thresholds).
│   └── Outputs: one 3-panel plot per raw file in ./Resultados/ and a summary table
│       (Resultados/resumen_picos.csv).
│
├── qc_emg.py
│   ├── What it does: Four-level quality control (recording, contraction, window, subject-leg).
│   │   Segments the valid contractions, extracts the 9 time/frequency features from 250 ms
│   │   windows, assigns the positional labels (first 25% = no fatigue, last 25% = fatigue)
│   │   and normalises the features to the subject-leg baseline.
│   └── Outputs: ./QC/ (qc_archivos.csv, qc_contracciones.csv, qc_sujetos.csv,
│       dataset_ventanas_completo.csv), review plots of the flagged/excluded recordings
│       (QC/revisar/) and QC/homogeneidad.png.
│
├── train_models.py
│   ├── What it does: Trains 10 machine-learning models with Leave-One-Subject-Out (LOSO)
│   │   validation, at two decision levels (window and contraction) and in three exclusion
│   │   scenarios (E0, E1, E2).
│   └── Outputs: results.csv, Modelos/ (resultados_completos.csv, tabla_comparacion.csv/.md,
│       tabla_por_clase.csv/.md, por_sujeto_mejor.csv, per-fold predictions in
│       Modelos/predicciones/) and quick-look plots.
│
├── report_analysis.py
│   ├── What it does: Report analyses: Table 6.1 with macro metrics and both baselines, failure
│   │   analysis (correlations with MDF decline, RMS change, SNR and cadence), bilateral leg
│   │   transfer, label-shift sensitivity and feature-group ablation.
│   └── Outputs: rep/resultados_reporte.json and rep/per_subject.csv.
│
├── generate_report_figures.py
│   ├── What it does: Generates every figure of the report from the outputs of the previous
│   │   scripts.
│   └── Outputs: rep/figs/ — fig1 (accepted vs. rejected trial), fig2 (feature trajectories),
│       fig3 (LOSO scheme), fig4 (confusion matrices), fig5 (macro-F1 per subject),
│       fig6 (feature ablation), figA1 (model comparison across scenarios),
│       figA2 (predicted probability by position), figA3 (risk–coverage curve).
│
├── EMG_RAW/
│   └── Raw data (not committed). One folder per subject with its CSV files.
│
├── results.csv
│   └── Results of every model and scenario (written by train_models.py).
│
├── requirements.txt
│   └── Python dependencies (numpy, pandas, scipy, matplotlib, scikit-learn).
│
└── README.md

## How to run

The raw recordings are not committed (submission rule). There are two ways
to reproduce the results, depending on whether the raw data is available.

### A. With the raw data (full pipeline)

Copy the class dataset into EMG_RAW/<subject>/*.csv and run the five
scripts in order:

    pip install -r requirements.txt

    export EMG_RAW=./EMG_RAW          # Windows PowerShell: $env:EMG_RAW = ".\EMG_RAW"
    export MPLBACKEND=Agg

    python process_emg.py
    python qc_emg.py
    python train_models.py
    python report_analysis.py
    python generate_report_figures.py

This reproduces every table and figure of the report (about 70 min in total,
most of it the 1D CNN; if PyTorch is installed the CNN runs faster, but its
results may differ slightly from the NumPy backend used in the report).

Each stage writes its artefacts into its own folder (Resultados/, QC/,
Modelos/, rep/) so that the next stage can read them.

### B. Without the raw data

The repository includes the derived files (QC/ window features,
Modelos/predicciones/ per-fold predictions, rep/), so every table and
figure can be rebuilt from them without access to the raw signals:

    python train_models.py
    python report_analysis.py
    python generate_report_figures.py

The classical models are retrained from the QC features if their predictions
are deleted. The CNN and Figure 1 need the raw signals, so without EMG_RAW/
their committed predictions and figure are reused.

## Settings

* Random seed: 42 (set in train_models.py, propagated to every
  scikit-learn estimator and to the CNN).

* CNN backend: train_models.py uses PyTorch when it is installed
  (e.g. Google Colab) and a NumPy + autograd implementation otherwise.
  The report was produced with the NumPy backend; to reproduce it exactly,
  set CNN_BACKEND=numpy (about 65 min on one CPU core).

* Environment variables:
    - EMG_RAW      — path to the folder with the raw CSVs. Defaults to ./EMG_RAW.
    - MPLBACKEND   — set to Agg for headless execution (no figure window).

