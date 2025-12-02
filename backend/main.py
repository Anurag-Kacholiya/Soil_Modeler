import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.base import clone
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# 🔹 Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px

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
        X_prep = prep_func(X)

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

def load_leaderboard():
    if LEADERBOARD_FILE.exists():
        try:
            df = pd.read_json(LEADERBOARD_FILE)
        except Exception as e:
            st.error(f"Failed to read leaderboard: {e}")
            return pd.DataFrame()
    return df


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
    """
    Train model using 5-fold CV and optionally RandomizedSearchCV for tuning.
    If hyperparams contain lists/tuples -> tuning is triggered.
    """

    # ---- Detect if tuning needed ----
    tuning_enabled = any(isinstance(v, (list, tuple)) for v in hyperparams.values())

    if tuning_enabled:
        base_model = model_builder()

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=hyperparams,
            n_iter=20,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            random_state=42
        )
        search.fit(X, y)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        # Normal fixed-parameter training
        model = model_builder(**hyperparams)
        model.fit(X, y)
        best_params = hyperparams

    # ---- 5-Fold OOF evaluation ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_trues = np.zeros(len(y))

    for tr, te in kf.split(X):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            oof_preds[te] = preds
            oof_trues[te] = y_test
        except Exception as e:
            st.warning(f"Fold failed: {e}")
            oof_preds[te] = np.nan
            oof_trues[te] = y_test

    # Metrics
    mask = ~np.isnan(oof_preds)
    metrics = calculate_metrics(oof_trues[mask], oof_preds[mask])

    # Final training on full data
    model.fit(X, y)

    return model, metrics, oof_trues, oof_preds, best_params


# ======================================================
# 🧩 Pipelines
# ======================================================
def run_full_pipeline(dataset_name, target_column_name, target_property_label,
                      progress_callback=None, update_progress=None):
    """
    Train all models with all preprocessing variants.
    Append results to leaderboard.json instead of overwriting.
    """
    X, y = load_data(dataset_name, target_column_name)
    if X is None:
        return pd.DataFrame()

    all_results = []
    total = len(PREPROCESSING_METHODS) * len(MODEL_CONFIG)
    count = 0

    for prep_key, prep_func in PREPROCESSING_METHODS.items():
        X_prep = prep_func(X)
        for model_name, cfg in MODEL_CONFIG.items():

            if progress_callback is not None:
                progress_callback(model_name=model_name, prep_key=prep_key)

            defaults = {p['name']: p['default'] for p in cfg['params']}

            try:
                model_builder = cfg.get('builder') or cfg.get('model')
                model, metrics, y_true, y_pred, best_params = train_model(
                    X_prep, y, model_builder, defaults
                )

                model_id = f"{dataset_name.split('.')[0]}_{target_property_label}_{prep_key}_{model_name}".lower()
                joblib.dump(model, MODELS_DIR / f"{model_id}.pkl")
                np.save(MODELS_DIR / f"{model_id}_y_true.npy", y_true)
                np.save(MODELS_DIR / f"{model_id}_y_pred.npy", y_pred)

                all_results.append({
                    'model_id': model_id,
                    'dataset': str(dataset_name),
                    'target': str(target_property_label),
                    'target_column': str(target_column_name),
                    'preprocessing_key': prep_key,
                    'preprocessing': prep_key,
                    'model': model_name,
                    **metrics,
                    'pickle_path': str(MODELS_DIR / f"{model_id}.pkl"),
                    'y_true_path': str(MODELS_DIR / f"{model_id}_y_true.npy"),
                    'y_pred_path': str(MODELS_DIR / f"{model_id}_y_pred.npy"),
                    'hyperparameters': best_params
                })

            except Exception as e:
                print(f"Failed to train {model_name} with {prep_key}: {e}")

            count += 1
            if update_progress is not None:
                update_progress(count / total)

    df = pd.DataFrame(all_results)

    existing = load_leaderboard()
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined.to_json(LEADERBOARD_FILE, orient='records', indent=2)

    return df



@st.cache_data(show_spinner="Retraining model...")
def run_single_pipeline(dataset_name, target_column_name, target_property_label, prep_key, model_name, hyperparameters):
    """Retrain a single tuned model (does not overwrite full leaderboard)."""
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
        model, metrics, y_true, y_pred, best_params = train_model(X_prep, y, model_builder, hyperparameters)

        model_id = f"{dataset_name.split('.')[0]}_{target_property_label}_{prep_key}_{model_name}_tuned".lower()
        joblib.dump(model, MODELS_DIR / f"{model_id}.pkl")
        np.save(MODELS_DIR / f"{model_id}_y_true.npy", y_true)
        np.save(MODELS_DIR / f"{model_id}_y_pred.npy", y_pred)
        return {
            'model_id': model_id,
            'dataset': str(dataset_name),
            'target': str(target_property_label),
            'target_column': str(target_column_name),
            'preprocessing_key': prep_key,
            'preprocessing': prep_key,
            'model': model_name,
            **metrics,
            'pickle_path': str(MODELS_DIR / f"{model_id}.pkl"),
            'y_true_path': str(MODELS_DIR / f"{model_id}_y_true.npy"),
            'y_pred_path': str(MODELS_DIR / f"{model_id}_y_pred.npy"),
            'hyperparameters': best_params
        }
    except Exception as e:
        st.error(f"Retraining failed: {e}")
        return None


# ======================================================
# 📊 Plotly Visualization Functions (Interactive)
# ======================================================
def plot_scatter_interactive(y_true, y_pred):
    """Interactive Predicted vs Actual scatter with identity line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(size=7, opacity=0.7),
        hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}",
        name="Predictions"
    ))

    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))

    fig.add_trace(go.Scatter(
        x=[min_v, max_v],
        y=[min_v, max_v],
        mode="lines",
        line=dict(dash="dash"),
        name="Ideal Line"
    ))

    fig.update_layout(
        title="Predicted vs Actual",
        template="plotly_dark",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        width=750,
        height=520
    )
    return fig


def plot_feature_importance_interactive(model, X, y, band_names):
    """
    Feature Importance Plot with Smoothing to reduce fluctuation noise.
    """

    # 1. Get raw importance scores
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importance (Spectral Regions)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        title = "Coefficient Importance (Spectral Regions)"
    else:
        # Lower repeats for speed in interactive mode
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = r.importances_mean
        title = "Permutation Importance (Spectral Regions)"

    # 2. Cleanup Data
    importances = np.asarray(importances, dtype=float).flatten()
    importances = np.nan_to_num(importances, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Create DataFrame
    n = min(len(importances), len(band_names))
    df_imp = pd.DataFrame({
        "Band_Label": band_names[:n],
        "Importance": importances[:n]
    })

    # 4. SCALING X-AXIS: Convert strings ("450", "452") to Numbers for proper scaling
    try:
        df_imp["Band_Numeric"] = pd.to_numeric(df_imp["Band_Label"])
        df_imp = df_imp.sort_values("Band_Numeric")
        x_col = "Band_Numeric"
        x_title = "Wavelength (nm)"
    except ValueError:
        # Fallback if bands represent names like "Band_1"
        df_imp["Band_Numeric"] = range(len(df_imp))
        x_col = "Band_Numeric"
        x_title = "Band Index"

    # 5. SMOOTHING
    window_size = 15 
    df_imp["Smoothed"] = df_imp["Importance"].rolling(window=window_size, center=True).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_imp[x_col],
        y=df_imp["Importance"],
        mode='lines',
        name='Raw Signal',
        line=dict(color='#6EA89E', width=1),
        opacity=0.3,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=df_imp[x_col],
        y=df_imp["Smoothed"],
        mode='lines',
        name='Trend',
        line=dict(color='#34D399', width=3),
        fill='tozeroy',
        fillcolor='rgba(52, 211, 153, 0.1)' 
    ))

    max_y = df_imp["Importance"].max()
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="#134E4A",
            zeroline=False,
            showspikes=True,
            spikethickness=1,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across"
        ),
        yaxis=dict(
            title="Importance Score",
            showgrid=True,
            gridcolor="#134E4A",
            zeroline=True,
            zerolinecolor="#34D399",
            range=[0, max_y * 1.1] 
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_wavelength_variance_interactive(X, band_names):
    """Variance of each wavelength band."""
    variances = X.var(axis=0)
    df = pd.DataFrame({"Band": band_names, "Variance": variances})

    fig = px.line(
        df,
        x="Band",
        y="Variance",
        title="Wavelength Variance",
        template="plotly_dark"
    )
    fig.update_layout(xaxis=dict(tickangle=45))
    return fig


def plot_property_correlation_interactive(X_prep, y_data, band_names):
    """Correlation between each band and the soil property."""
    corr = np.array([
        np.corrcoef(X_prep.iloc[:, i], y_data)[0, 1] if np.std(X_prep.iloc[:, i]) > 0 else 0
        for i in range(X_prep.shape[1])
    ])
    df = pd.DataFrame({"Band": band_names, "Correlation": corr})

    fig = px.line(
        df,
        x="Band",
        y="Correlation",
        title="Property vs Wavelength Correlation",
        template="plotly_dark"
    )
    fig.update_layout(xaxis=dict(tickangle=45))
    fig.add_hline(y=0, line_dash="dash")
    return fig


def plot_spectral_profiles_interactive(X_prep, band_names=None, preprocess_key="reflectance", n_samples=12):
    if band_names is None:
        band_names = list(X_prep.columns)

    y_label_map = {
        "reflectance": "Reflectance",
        "absorbance": "Absorbance (Log 1/R)",
        "continuumremoval": "Continuum-Removed Signal"
    }

    y_label = y_label_map.get(preprocess_key.lower(), "Processed Value")

    n_samples = min(n_samples, len(X_prep))
    indices = np.random.choice(len(X_prep), n_samples, replace=False)

    fig = go.Figure()
    for idx in indices:
        fig.add_trace(go.Scatter(
            x=list(range(len(band_names))),       # numeric x axis
            y=X_prep.iloc[idx, :],
            mode="lines",
            opacity=0.65,
            name=f"Sample {idx}"
        ))

    fig.update_layout(
        title=f"Spectral Profiles ({preprocess_key})",
        xaxis_title="Wavelength (Bands)",
        yaxis_title=y_label,
        template="plotly_dark"
    )

    return fig



def plot_band_accuracy_curve_interactive(model, X_prep, y, band_names):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        r = permutation_importance(model, X_prep, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = r.importances_mean

    importances = np.asarray(importances, dtype=float).flatten()
    importances = np.nan_to_num(importances)

    order = np.argsort(importances)[::-1]
    n_features = X_prep.shape[1]

    k_values = [k for k in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100] if k <= n_features]

    records = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for k in k_values:
        idx = order[:k]
        idx = idx[idx < n_features]

        if len(idx) == 0:
            continue

        X_sub = X_prep.iloc[:, idx]

        fold_scores = []
        for tr, te in kf.split(X_sub):
            X_tr, X_te = X_sub.iloc[tr], X_sub.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            mdl = clone(model)
            try:
                mdl.fit(X_tr, y_tr)
                preds = mdl.predict(X_te)
                fold_scores.append(r2_score(y_te, preds))
            except:
                continue

        if fold_scores:
            records.append({"Bands": k, "R2": float(np.mean(fold_scores))})

    if not records:
        fig = px.line(title="R² vs Number of Bands")
        fig.update_layout(template="plotly_dark")
        return fig

    df_curve = pd.DataFrame(records)
    fig = px.line(df_curve, x="Bands", y="R2", markers=True,
                  title="R² vs Number of Bands (Top-k Bands)", template="plotly_dark")
    fig.update_layout(xaxis_title="Bands", yaxis_title="R²")
    return fig

# ======================================================
# 🚀 Streamlit Routing
# ======================================================
def main():
    from frontend.app_page import show_app_page

    st.set_page_config(
        page_title="Spectral Soil Modeler",
        layout="wide",
    )

    show_app_page()


if __name__ == "__main__":
    main()
