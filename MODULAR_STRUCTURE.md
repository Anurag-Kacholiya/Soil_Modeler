# Spectral Soil Modeler - Modular Structure

This document explains how the monolithic `main.py` file has been reorganized into a clean, modular structure.

## 📁 Project Structure

```
SSD_Final_project/
├── main.py                    # Main app entry point (simplified)
├── backend/
│   ├── __init__.py           # Backend package initialization
│   └── main.py               # Core ML functions and data processing
├── frontend/
│   ├── __init__.py           # Frontend package initialization
│   ├── landing_page.py       # Landing page UI components
│   ├── results_page.py       # Results and tuning UI components
│   └── visualization_page.py # Visualization dashboard UI components
├── models/
│   ├── __init__.py           # Model configurations and imports
│   ├── pls_model.py          # PLSR model implementation
│   ├── cubist_model.py       # Cubist model implementation
│   ├── gbrt_model.py         # Gradient Boosting model implementation
│   ├── krr_model.py          # Kernel Ridge Regression model implementation
│   └── svr_model.py          # Support Vector Regression model implementation
├── preprocessing/
│   ├── __init__.py           # Preprocessing package initialization
│   ├── reflectance.py        # Raw reflectance preprocessing
│   ├── absorbance.py         # Absorbance transformation
│   └── continuum_removal.py  # Continuum removal preprocessing
├── dataset/                  # Data files directory
├── models_store/            # Trained model storage
└── requirements.txt         # Python dependencies
```

## 🔄 What Was Reorganized

### 1. **Backend (`backend/main.py`)**
Contains all the core machine learning functionality:
- `get_available_datasets()` - Dataset discovery
- `load_data()` - Data loading and validation  
- `preprocess_data()` - Spectral preprocessing methods
- `get_model()` - Model initialization with hyperparameters
- `calculate_metrics()` - Performance metrics (R², RMSE, RPD)
- `train_model()` - 5-fold cross-validation training
- `run_full_pipeline()` - Complete analysis pipeline (15 models)
- `run_single_pipeline()` - Individual model retraining
- `plot_scatter()` - Predicted vs Actual plots
- `plot_feature_importance()` - Feature importance visualization

### 2. **Frontend (`frontend/`)**
Split into three focused UI modules:

**`landing_page.py`:**
- Dataset selection interface
- Target column selection
- Property label selection  
- Analysis execution controls
- Previous results loading

**`results_page.py`:**
- Leaderboard display with sorting
- Hyperparameter tuning forms
- Model retraining interface
- Navigation to visualization

**`visualization_page.py`:**
- Performance metrics display
- Predicted vs Actual scatter plots
- Feature importance plots
- Results download functionality

### 3. **Models (`models/`)**
Centralized model configurations:
- `MODEL_CONFIG` dictionary with all hyperparameter definitions
- Individual model implementation files (extensible)
- Type definitions (int, float, select) for UI generation

### 4. **Preprocessing (`preprocessing/`)**
Modular preprocessing functions:
- `apply_reflectance()` - Raw data (no transformation)
- `apply_absorbance()` - -log10(R) transformation  
- `apply_continuum_removal()` - Convex hull normalization

### 5. **Main App (`main.py`)**
Simplified to only handle:
- Streamlit configuration
- Session state initialization  
- Page routing between frontend modules
- Import coordination

## ✅ Benefits of This Structure

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Maintainability**: Easy to find and modify specific functionality
3. **Extensibility**: Add new models/preprocessing methods without touching other code
4. **Testing**: Individual components can be tested in isolation
5. **Reusability**: Backend functions can be imported by other scripts
6. **Collaboration**: Multiple developers can work on different modules simultaneously

## 🚀 How to Run

The app works exactly the same as before:

```bash
cd /Users/bharadwaj/Desktop/SSD_Final_project
/Users/bharadwaj/Desktop/SSD_Final_project/soil_env/bin/python -m streamlit run main.py
```

The modular structure is completely transparent to users - all functionality remains identical while the code is now much more organized and maintainable.

## 🔧 Key Configuration

- **Models**: Defined in `models/__init__.py` with complete hyperparameter specifications
- **Preprocessing**: Each method isolated in `preprocessing/` folder
- **Data Flow**: `backend/main.py` → `frontend/` modules → `main.py` routing
- **State Management**: All session state handled in simplified `main.py`

This structure follows Python best practices and makes the project ready for future expansion and team development.
