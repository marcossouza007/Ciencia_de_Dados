# Ciência de Dados

Academic projects and professional studies focused on **Data Science**, **Machine Learning**,
and **LLM applications**.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [LLM Applications](#llm-applications)
- [Notebooks](#notebooks)
- [Getting Started](#getting-started)
- [Documentation](#documentation)

---

## Project Structure

```
.
├── data/
│   ├── raw/            # Original, immutable source data
│   └── processed/      # Cleaned and feature-engineered datasets
├── models/             # Serialised trained models (.joblib / .pkl)
├── notebooks/          # Google Colab–compatible Jupyter notebooks
│   ├── eda_exploration.ipynb
│   ├── ml_model_training.ipynb
│   ├── anomaly_detection.ipynb
│   └── llm_applications.ipynb
├── scripts/
│   ├── eda/            # Data cleaning, preprocessing, EDA
│   ├── ml/             # Feature engineering, training, evaluation
│   ├── anomaly_detection/  # Network & system log anomaly detection
│   └── llm/            # LLM-powered NLP applications
├── docs/               # Per-module documentation
├── requirements.txt
└── README.md
```

---

## Core Components

### EDA (`scripts/eda/`)

| Module | Purpose |
|---|---|
| `data_cleaning.py` | Missing-value imputation, duplicate removal, outlier clipping |
| `preprocessing.py` | Encoding, scaling, train/val/test splitting |
| `exploratory_analysis.py` | Distributions, correlations, automated EDA report |

→ [EDA Documentation](docs/eda/README.md)

### Machine Learning (`scripts/ml/`)

| Module | Purpose |
|---|---|
| `feature_engineering.py` | Polynomial features, datetime decomposition, binning, target encoding |
| `model_training.py` | Cross-validated training for Logistic Regression, Random Forest, Gradient Boosting |
| `model_evaluation.py` | Metrics, confusion matrix, ROC/PR curves, feature importances |

→ [ML Documentation](docs/ml/README.md)

### Anomaly Detection (`scripts/anomaly_detection/`)

| Module | Purpose |
|---|---|
| `network_anomaly_detector.py` | Log feature extraction, Isolation Forest, LOF, `AnomalyDetector` class |

→ [Anomaly Detection Documentation](docs/anomaly_detection/README.md)

---

## LLM Applications (`scripts/llm/`)

| Module | Purpose |
|---|---|
| `text_classification.py` | Zero-shot / few-shot text classification via OpenAI API |
| `text_summarization.py` | Single-pass and map-reduce document summarisation |
| `log_analysis.py` | Log triage, incident summarisation, root-cause analysis |

→ [LLM Documentation](docs/llm/README.md)

---

## Notebooks

All notebooks under `notebooks/` are designed to run in **Google Colab** with minimal setup.

| Notebook | Description |
|---|---|
| `eda_exploration.ipynb` | Walkthrough of cleaning, preprocessing, and EDA visualisations |
| `ml_model_training.ipynb` | Feature engineering, CV training, hyper-parameter tuning, evaluation |
| `anomaly_detection.ipynb` | Log feature extraction and anomaly detection with IF and LOF |
| `llm_applications.ipynb` | Text classification, summarisation, and log analysis with OpenAI |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/marcossouza007/Ci-ncia_de_Dados.git
cd Ci-ncia_de_Dados

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set your OpenAI API key for LLM scripts
export OPENAI_API_KEY="sk-..."

# 5. Run an EDA demo
python scripts/eda/exploratory_analysis.py
```

---

## Documentation

Detailed documentation for each module lives in the `docs/` directory:

- [`docs/eda/README.md`](docs/eda/README.md)
- [`docs/ml/README.md`](docs/ml/README.md)
- [`docs/anomaly_detection/README.md`](docs/anomaly_detection/README.md)
- [`docs/llm/README.md`](docs/llm/README.md)
