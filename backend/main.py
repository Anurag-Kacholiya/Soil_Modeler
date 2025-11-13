import sys
import os

# --- Add project root to path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# --- Import modular configs ---
from models import MODEL_CONFIG
from preprocessing import PREPROCESSING_METHODS


# ======================================================
# 🔧 Preprocessing Wrapper
# ======================================================
def preprocess_data(X, method_key: str):
    """
    Wrapper to apply preprocessing dynamically.
    Used by visualization_page.py and pipeline.
    """
    if method_key not in PREPROCESSING_METHODS:
        raise ValueError(f"❌ Unknown preprocessing method: {method_key}")

    try:
        prep_func = PREPROCESSING_METHODS[method_key]
        st.write(f"🧪 Applying preprocessing: {method_key}")
        X_prep = prep_func(X)

        # Ensure DataFrame output
        if not isinstance(X_prep, pd.DataFrame):
            X_prep = pd.DataFrame(X_prep, columns=X.columns)

        st.write(f"✅ Preprocessing complete: {X_prep.shape[0]} samples × {X_prep.shape[1]} features")
        return X_prep

    except Exception as e:
        st.error(f"⚠️ Error during preprocessing ({method_key}): {e}")
        return X


# ======================================================
# 📂 Directory Setup
# ======================================================
DATA_DIR = Path('dataset')
MODELS_DIR = Path('models_store')
LEADERBOARD_FILE = Path('leaderboard.json')

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ======================================================
# 📘 Dataset Loader
# ======================================================
def get_available_datasets():
    """Return CSV/XLS/XLSX files in dataset/"""
    files = list(DATA_DIR.glob('*.csv')) + list(DATA_DIR.glob('*.xls')) + list(DATA_DIR.glob('*.xlsx'))
    return [f.name for f in files]


def load_data(dataset_name, target_col):
    """Load dataset and split into X, y"""
    path = DATA_DIR / dataset_name
    df = pd.read_csv(path)

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        return None, None

    y = df[target_col].reset_index(drop=True)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    st.session_state.band_names = list(X.columns)
    return X.reset_index(drop=True), y


# ======================================================
# 🧮 Model Training + Evaluation
# ======================================================
def calculate_metrics(y_true, y_pred):
    """Compute R², RMSE, and RPD metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse if rmse != 0 else np.nan
    return {'r2': r2, 'rmse': rmse, 'rpd': rpd}


def train_model(X, y, model_builder, hyperparams):
    """Perform 5-fold cross-validation and retrain final model"""
    # ✅ FIX: unpack dictionary of parameters
    try:
        model = model_builder(**hyperparams)
    except TypeError:
        # fallback if builder expects dict
        model = model_builder(hyperparams)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_trues = np.zeros(len(y))

    for tr, te in kf.split(X):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        try:
            model.fit(X_train, y_train)
            oof_preds[te] = model.predict(X_test)
            oof_trues[te] = y_test
        except Exception as e:
            oof_preds[te] = np.nan
            oof_trues[te] = y_test
            st.warning(f"Fold failed: {e}")

    mask = ~np.isnan(oof_preds)
    metrics = calculate_metrics(oof_trues[mask], oof_preds[mask]) if mask.any() else {'r2': np.nan, 'rmse': np.nan, 'rpd': np.nan}

    # Retrain full model
    model.fit(X, y)
    return model, metrics, oof_trues, oof_preds


# ======================================================
# 🧩 Pipelines
# ======================================================
@st.cache_data(show_spinner="Running full analysis...")
def run_full_pipeline(dataset_name, target_column_name, target_property_label):
    """Train all models with all preprocessing variants"""
    X, y = load_data(dataset_name, target_column_name)
    if X is None:
        return pd.DataFrame()

    all_results = []
    progress_bar = st.progress(0.0)
    total = len(PREPROCESSING_METHODS) * len(MODEL_CONFIG)
    count = 0

    for prep_key, prep_func in PREPROCESSING_METHODS.items():
        X_prep = prep_func(X)
        for model_name, cfg in MODEL_CONFIG.items():
            defaults = {p['name']: p['default'] for p in cfg['params']}
            try:
                st.text(f"Training {model_name} ({prep_key})...")
                model_builder = cfg.get('builder') or cfg.get('model')
                model, metrics, y_true, y_pred = train_model(X_prep, y, model_builder, defaults)

                model_id = f"{dataset_name.split('.')[0]}_{target_property_label}_{prep_key}_{model_name}".lower()
                joblib.dump(model, MODELS_DIR / f"{model_id}.pkl")
                np.save(MODELS_DIR / f"{model_id}_y_true.npy", y_true)
                np.save(MODELS_DIR / f"{model_id}_y_pred.npy", y_pred)

                all_results.append({
                    'model_id': model_id,
                    'dataset': dataset_name,
                    'target': target_property_label,
                    'target_column': target_column_name,
                    'preprocessing_key': prep_key,
                    'preprocessing': prep_key,
                    'model': model_name,
                    **metrics,
                    'pickle_path': str(MODELS_DIR / f"{model_id}.pkl"),
                    'y_true_path': str(MODELS_DIR / f"{model_id}_y_true.npy"),
                    'y_pred_path': str(MODELS_DIR / f"{model_id}_y_pred.npy"),
                    'hyperparameters': defaults
                })
            except Exception as e:
                st.warning(f"Failed to train {model_name} with {prep_key}: {e}")
            count += 1
            progress_bar.progress(count / total)

    progress_bar.empty()
    df = pd.DataFrame(all_results)
    df.to_json(LEADERBOARD_FILE, orient='records', indent=2)
    return df


@st.cache_data(show_spinner="Retraining model...")
def run_single_pipeline(dataset_name, target_column_name, target_property_label, prep_key, model_name, hyperparameters):
    """Retrain a single tuned model"""
    X, y = load_data(dataset_name, target_column_name)
    if X is None:
        return None

    prep_func = PREPROCESSING_METHODS.get(prep_key)
    if prep_func is None:
        st.error(f"Unknown preprocessing method: {prep_key}")
        return None

    X_prep = prep_func(X)
    try:
        model_builder = MODEL_CONFIG[model_name].get('builder') or MODEL_CONFIG[model_name]['model']
        model, metrics, y_true, y_pred = train_model(X_prep, y, model_builder, hyperparameters)

        model_id = f"{dataset_name.split('.')[0]}_{target_property_label}_{prep_key}_{model_name}_tuned".lower()
        joblib.dump(model, MODELS_DIR / f"{model_id}.pkl")
        np.save(MODELS_DIR / f"{model_id}_y_true.npy", y_true)
        np.save(MODELS_DIR / f"{model_id}_y_pred.npy", y_pred)
        return {
            'model_id': model_id,
            'dataset': dataset_name,
            'target': target_property_label,
            'target_column': target_column_name,
            'preprocessing_key': prep_key,
            'preprocessing': prep_key,
            'model': model_name,
            **metrics,
            'pickle_path': str(MODELS_DIR / f"{model_id}.pkl"),
            'y_true_path': str(MODELS_DIR / f"{model_id}_y_true.npy"),
            'y_pred_path': str(MODELS_DIR / f"{model_id}_y_pred.npy"),
            'hyperparameters': hyperparameters
        }
    except Exception as e:
        st.error(f"Retraining failed: {e}")
        return None


# ======================================================
# 📊 Visualization Functions
# ======================================================
def plot_scatter(y_true, y_pred, title="Predicted vs Actual"):
    fig, ax = plt.subplots(figsize=(6, 6))
    mask = ~np.isnan(y_pred)
    if mask.any():
        sns.scatterplot(x=y_true[mask], y=y_pred[mask], ax=ax, alpha=0.7)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True)
    return fig


def plot_feature_importance(model, X, y, band_names=None):
    """
    Generate and display feature importance for a trained model.
    Works for tree-based, linear, or other sklearn-compatible models.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.inspection import permutation_importance

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Get feature importance depending on model type ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        title = "Coefficient Importance"
    else:
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = r.importances_mean
        title = "Permutation Importance"

    # --- Convert to numeric 1D array ---
    importances = np.asarray(importances, dtype=float).flatten()
    importances = np.nan_to_num(importances)

    # --- Align with band names ---
    if band_names is None:
        band_names = list(X.columns)
    n = min(len(importances), len(band_names))
    importances, band_names = importances[:n], band_names[:n]

    # --- Create DataFrame safely ---
    imp_df = pd.DataFrame({
        "band": band_names,
        "importance": importances.astype(float)
    })

    # --- Sort and plot ---
    imp_df = imp_df.sort_values("importance", ascending=False).head(30)
    sns.barplot(x="importance", y="band", data=imp_df, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Spectral Band")
    ax.grid(True)

    return fig



def plot_wavelength_variance(X, band_names=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    variances = X.var(axis=0)
    labels = band_names if band_names else list(X.columns)
    ax.plot(labels, variances, lw=2)
    ax.set_title("📈 Wavelength Variance")
    ax.set_xlabel("Wavelength (Bands)")
    ax.set_ylabel("Variance")
    ax.grid(True)
    return fig


def plot_property_wavelength_relationship(X, y, band_names=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = band_names if band_names else list(X.columns)
    corr = [np.corrcoef(X[col], y)[0, 1] if np.std(X[col]) > 0 else 0 for col in X.columns]
    ax.plot(labels, corr, lw=2)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("🔍 Property-Wavelength Relationship")
    ax.set_xlabel("Wavelength (Bands)")
    ax.set_ylabel("Correlation")
    ax.grid(True)
    return fig


def plot_spectral_profiles(X, y=None, band_names=None, n_samples=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_samples = min(n_samples, len(X))
    rng = np.random.default_rng(42)
    for i in rng.choice(len(X), n_samples, replace=False):
        ax.plot(band_names if band_names else list(X.columns), X.iloc[i, :], alpha=0.6)
    ax.set_title("🌈 Spectral Profiles of Random Samples")
    ax.set_xlabel("Wavelength (Bands)")
    ax.set_ylabel("Reflectance")
    ax.grid(True)
    return fig


# ======================================================
# 🚀 Streamlit Routing
# ======================================================
def main():
    from frontend.landing_page import show_landing_page
    from frontend.results_page import show_results_page
    from frontend.visualization_page import show_visualization_page

    st.sidebar.title("🌱 Spectral Soil Modeler")

    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "results":
        show_results_page()
    elif st.session_state.page == "visualize":
        show_visualization_page()


if __name__ == "__main__":
    main()
