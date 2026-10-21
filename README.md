# EMG-based detection of quadriceps fatigue — Group 01

Surface EMG pipeline for the detection and classification of acute peripheral
fatigue in the quadriceps during repeated unilateral knee extensions. The
repository implements the full analysis reported in the manuscript: signal
processing, four-level quality control, feature extraction, within-set
labelling, leave-one-subject-out (LOSO) classification, and every figure
in the report.


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

