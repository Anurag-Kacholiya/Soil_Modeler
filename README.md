# 🌱 Spectral Soil Modeler  
### *A Streamlit-based Machine Learning System for Hyperspectral Soil Property Prediction*  
### **Team 35 — SSD Final Project**

---

## 📌 Overview

**Spectral Soil Modeler** is an interactive machine learning application designed to predict soil properties (e.g., clay, organic carbon, nutrients, moisture) from **hyperspectral reflectance data**.

It replaces slow, destructive laboratory tests with a fast, non-destructive, and scalable spectral-ML pipeline. Users can upload/select datasets, apply preprocessing, run model pipelines, inspect a dynamic leaderboard, retrain models, and explore interactive visual diagnostics.

The project is implemented as a unified Streamlit frontend with a modular Python backend.

---

## 📁 Corrected Project Structure

```

Soil_Modeler/
├── app.py                    # Main Streamlit entry point (UI + navigation)
│
├── backend/
│   ├── **init**.py
│   └── main.py               # Core ML pipeline: loading, preprocessing, training, metrics, plotting helpers
│
├── models/
│   ├── **init**.py
│   ├── pls_model.py          # Partial Least Squares Regression
│   ├── cubist_model.py       # Cubist-style model
│   ├── gbrt_model.py        # Gradient Boosting Regressor
│   ├── krr_model.py         # Kernel Ridge Regressor
│   └── svr_model.py          # Support Vector Regressor
│
├── preprocessing/
│   ├── **init**.py
│   ├── reflectance.py        # Raw reflectance
│   ├── absorbance.py         # -log10(R) transformation
│   └── continuum_removal.py  # Convex hull normalization
│
├── frontend/
│   ├── components/
│   │    ├── center_panel.py      # Model Results (main center panel)
│   │    ├── leaderboard_panel.py # Leaderboard (sidebar)
│   │    └── right_panel.py       # Diagnostics (right panel)
│   ├── **init**.py               # Frontend package initialization
│   ├── landing_page.py           # Landing page UI components
│   └── app_page.py               # Results page (combined Model Results + Diagnostics)
│
├── dataset/                      # Input spectral datasets (CSV / XLS)
├── models_store/                 # Saved trained ML models (joblib / pickle)
├── leaderboard.json              # Persistent leaderboard state
└── requirements.txt

````

> **Notes:**  
> - `app.py` is the Streamlit entrypoint; it orchestrates navigation and imports the `frontend` package pages (`landing_page.py`, `app_page.py`).  
> - `backend/main.py` contains the pipeline and exposes functions used by both frontend and backend modules.

---

## 🔥 Key Features

- **Two-page workflow** (Landing → Results) with a dynamic sidebar leaderboard.  
- **Interactive visualizations** (Plotly) — Predicted vs Measured, Feature Importance, Band Sensitivity, Raw Spectra.  
- **Per-model retraining** with editable hyperparameters (updates leaderboard in real time).  
- **Modular codebase**: easy to add models or preprocessing functions.  
- **Persistent leaderboard** using `leaderboard.json` so rankings survive restarts.

---

## 🧠 Backend Overview (`backend/main.py`)

Main responsibilities:

1. **Data loading** — utilities to list and load datasets from `dataset/`.  
2. **Preprocessing** — wrappers that call `preprocessing/*.py` functions to produce model-ready data.  
3. **Model orchestration** — functions to build, train, predict for each model in `models/`.  
4. **Evaluation metrics** — R², RMSE, RPD, residual diagnostics.  
5. **Pipeline execution** — `run_full_pipeline()` (runs all configured models) and `run_single_pipeline()` (for per-model retrain).  
6. **Visualization helpers** — returns data for Plotly charts (center/right panels).

---

## 🎨 Frontend Layout (`frontend/`)

The frontend is split into reusable UI components under `frontend/components/` and page logic under `landing_page.py` and `app_page.py`.

### Landing Page (`landing_page.py`)
- Dataset selection & preview
- Target property selection
- Preprocessing choice
- Buttons to run full pipeline or run a single model

### Results / App Page (`app_page.py`)
- **Sidebar**: `leaderboard_panel.py` — dynamic leaderboard displaying model rank (R², RMSE, RPD)
- **Center panel**: `center_panel.py` — Model Results (Predicted vs Measured, Feature Importance, Model Config, Retrain form)
- **Right panel**: `right_panel.py` — Diagnostics (Raw Spectra, Band Sensitivity)
- Retraining in the center panel updates model metrics and writes to `leaderboard.json`

---

## 🔬 Models (`models/`)

Each model file exposes a consistent interface (for integration with `backend/main.py` and the frontend):

- `build_model(hyperparams)`  
- `train(X_train, y_train)`  
- `predict(X)`  
- `get_feature_importance()` (where available)

Models included:
- **PLSR** — Partial Least Squares Regression, widely used in spectroscopy.  
- **Cubist-style** — Rule-based, interpretable regression.  
- **GBRT** — Gradient Boosting Regressor for non-linear patterns.  
- **KRR** — Kernel Ridge Regression for smooth non-linear mapping.  
- **SVR** — Support Vector Regression.

All model hyperparameters are surfaced to the UI for tuning.

---

## ⚙️ Preprocessing (`preprocessing/`)

- `reflectance.py` — raw reflectance handling and normalization.  
- `absorbance.py` — converts reflectance to absorbance using `-log10(R)`.  
- `continuum_removal.py` — convex-hull continuum removal to emphasize absorption features.

Preprocessing modules return transformed DataFrames ready for model training and plotting.

---

## 📦 leaderboard.json

`leaderboard.json` stores persistent leaderboard entries (model name, metrics, metadata, timestamp). The app reads from and writes to this file when models are trained or retrained so leaderboard ranking persists across restarts.

---

## 🚀 How to Run

1. Clone repository:
```bash
git clone https://github.com/Anurag-Kacholiya/Soil_Modeler
cd Soil_Modeler
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the deployed app (if needed):
   `https://soilmodeler.streamlit.app/`

---

## 🧭 Typical User Workflow

1. Launch `app.py` → Landing page loads.
2. Select dataset and target property.
3. Choose preprocessing method (Reflectance / Absorbance / Continuum Removal).
4. Preview spectral data and run pipeline.
5. Results page opens with leaderboard in sidebar.
6. Select a model to inspect center panel (Pred vs Measured / Feature Importance).
7. Use right panel for Diagnostics (Raw Spectra, Band Sensitivity).
8. Retrain models from the center panel — updated metrics persist to `leaderboard.json`.

---

## 🧾 Why This Design

* **Accurate & fast**: supports rapid, non-destructive soil property estimation via hyperspectral data.
* **User-centric**: streamlined two-page UI with clear separation of dataset setup and result exploration.
* **Reproducible**: modular backend and persistent leaderboard enable repeatable experiments.
* **Extensible**: new models, preprocessing steps, or visual components can be added with minimal changes.

---

## 🙌 Contributors (Team 35)

* **Anurag Kacholiya** (2025202025) — Preprocessing, documentation, general integration
* **V. S. S. Bharadwaja** (2025204012) — Backend engineering, pipeline logic, testing
* **Afzal Basha Shaik** (2025201097) — Frontend visualizations, Plotly integration, `app_page.py` components
* **Prabhash Pradhan** (2025201089) — ML models, hyperparameter tuning
* **Aringi Vinay Chaitanya** (2025201041) — Landing page, frontend structure, `app.py` integration
