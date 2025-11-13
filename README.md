# 🌱 **Spectral Soil Modeler**

### *A Streamlit-based Machine Learning System for Soil Spectral Analysis*

### **Team 35 — SSD Final Project**

---

## 📌 Overview

**Spectral Soil Modeler** is an interactive machine learning application designed to predict soil properties using **hyperspectral reflectance data**.

Users can upload spectral datasets, apply preprocessing transformations, train regression models, compare performance, visualize results, and analyze spectral characteristics.

This project was originally planned as a **FastAPI + MERN stack** system in Phase-1, but was redesigned into a **unified Streamlit + modular Python architecture** in Phase-2 for simplicity, speed, and better ML workflow integration.

---

# 📁 Project Structure

```
SSD_Final_project/
├── backend/
│   ├── __init__.py
│   └── main.py               # Streamlit entry point (routing + UI control) & Core ML pipeline: loading, preprocessing, training, metrics, plots
├── frontend/
│   ├── __init__.py
│   ├── landing_page.py       # Page 1 – dataset input & pipeline execution
│   ├── results_page.py       # Page 2 – leaderboard + hyperparameter tuning
│   └── visualization_page.py # Page 3 – Visual dashboards & plots
├── models/
│   ├── __init__.py           # MODEL_CONFIG + hyperparameter metadata
│   ├── pls_model.py          # Partial Least Squares Regression
│   ├── cubist_model.py       # Cubist-like model
│   ├── gbrt_model.py         # Gradient Boosting Regressor
│   ├── krr_model.py          # Kernel Ridge Regressor
│   └── svr_model.py          # Support Vector Regressor
├── preprocessing/
│   ├── __init__.py
│   ├── reflectance.py        # Raw reflectance / no transformation
│   ├── absorbance.py         # -log10(R) transformation
│   └── continuum_removal.py  # Convex hull normalization
├── dataset/                  # Input spectral datasets (XLS/CSV)
├── models_store/             # Saved trained ML models
└── requirements.txt          # Python dependencies
```

---

# 🔥 Key Features

### ✔ Interactive Streamlit UI

Three clean, modular pages:

1. **Landing Page** – Data selection, preprocessing selection, and pipeline execution
2. **Results Page** – Model leaderboard, tuning, retraining
3. **Visualization Page** – Scatter plots, feature importance, wavelength variance, and more

---

# 🧠 Backend Architecture (`backend/main.py`)

The backend contains all core ML operations in one centralized module:

### **1. Dataset Management**

* `get_available_datasets()`
* `load_data()`

### **2. Preprocessing**

Applies transformations imported from `preprocessing/`:

* `apply_reflectance()`
* `apply_absorbance()`
* `apply_continuum_removal()`

### **3. Model Initialization**

* `get_model(model_name, hyperparams)`

### **4. Metrics Calculation**

* R² (Coefficient of Determination)
* RMSE (Root Mean Squared Error)
* RPD (Residual Predictive Deviation)

### **5. Training Pipelines**

* `train_model()`
* `run_full_pipeline()` → trains all 15 hyperparameter combinations
* `run_single_pipeline()` → for targeted retraining

### **6. Visualization Utilities**

* `plot_scatter()` – Pred vs Actual
* `plot_feature_importance()` – GBRT, Cubist, PLS
* Additional spectral visualizations handled in frontend

---

# 🎨 Frontend Architecture (`frontend/`)

## **1. Landing Page**

Handles:

* Dataset selection
* Identifying target property column
* Choosing preprocessing method
* Running full or single-model pipeline
* Dataset preview

---

## **2. Results Page**

Displays:

* Leaderboard table sorted by model metrics
* Hyperparameter tuning forms auto-generated from MODEL_CONFIG
* Buttons to retrain each model
* Option to navigate to visualization

---

## **3. Visualization Page**

Provides comprehensive spectral and model visualizations:

### **Model-Based Visualizations**

* Predicted vs Actual scatter plot
* Feature importance (GBRT, Cubist, PLSR)

### **Spectral Analysis Visualizations**

* **Wavelength Variance Plot**
  Shows variability across wavelengths in the spectral dataset
* **Property–Wavelength Relationship**
  Explores correlation between target property and spectral features
* **Spectral Profiles**
  Visualizes reflectance/absorbance for selected samples

---

# 🔬 Models (`models/`)

Each model is defined in its own file with:

* `build_model()`
* Training and prediction logic
* Hyperparameters handled via `MODEL_CONFIG`

Models included:

| Model       | Description                               |
| ----------- | ----------------------------------------- |
| PLSR        | Linear regression using latent components |
| Cubist-like | Tree rule-based regression                |
| GBRT        | Gradient Boosting                         |
| KRR         | Kernel Ridge Regression                   |
| SVR         | Support Vector Regression                 |

All hyperparameters are exposed to the UI through structured configuration dictionaries.

---

# ⚙️ Preprocessing (`preprocessing/`)

| Method            | Description                                  |
| ----------------- | -------------------------------------------- |
| Reflectance       | Uses raw reflectance values                  |
| Absorbance        | -log10(R) transformation                     |
| Continuum Removal | Convex hull normalization for spectral shape |

Each preprocessing method returns a transformed DataFrame ready for model training.

---

# 🚀 Running the Application

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Run the Streamlit app**

```
streamlit run main.py
```

---

# 🧭 User Workflow Summary

1. Open app → select dataset
2. Choose preprocessing + target property
3. Run full pipeline or single model
4. View results + tune hyperparameters
5. Navigate to visualization page for insights
6. Download results or trained models

---

# 🧾 Benefits of This Modular Architecture

* **Separation of concerns**
  UI, models, preprocessing, and backend are fully isolated.

* **Scalability**
  New models or preprocessing methods can be added easily.

* **Maintainability**
  Clear folder separation enables easier navigation.

* **Testing**
  Individual modules can be unit-tested independently.

* **Developer collaboration**
  Multiple team members can work on separate modules without conflicts.

---

# 🧩 Technologies Used

* **Python 3.10+**
* **Streamlit** (UI)
* **Pandas / NumPy** (Data handling)
* **Scikit-learn** (Machine Learning)
* **Matplotlib** (Plots)
* **Joblib** (Model persistence)

---

# 🎯 Conclusion

This project provides a complete end-to-end solution for soil spectral modeling — from dataset loading and preprocessing to model training, evaluation, and visualization.

The Phase-2 redesign resulted in a **cleaner, modular, efficient system** highly suitable for ML experimentation and spectral analysis.

---

# 🙌 Contributors

**Team 35 – SSD Course Project**
- Anurag Kacholiya - 2025202025
- Vadali S S Bharadwaja - 2025204012
- Afzal Basha Shaik - 2025201097
- Prabhash Pradhan - 2025201089
- Aringi Vinay Chaitanya - 2025201041