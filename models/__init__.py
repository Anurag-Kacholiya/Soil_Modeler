"""
Models package for Spectral Soil Modeler.
Contains definitions for all regression algorithms.
"""
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from cubist import Cubist

from .pls_model import get_plsr_model
from .cubist_model import get_cubist_model
from .gbrt_model import get_gbrt_model
from .krr_model import get_krr_model
from .svr_model import get_svr_model

MODEL_CONFIG = {
    'PLSR': {
        'model': PLSRegression,
        'params': [
            {'name': 'n_components', 'type': 'int', 'min': 1, 'max': 50, 'default': 10}
        ]
    },
    'Cubist': {
        'model': Cubist,
        'params': [{'name': 'n_rules', 'type': 'int', 'min': 1, 'max': 1000, 'default': 100}]
    },
    'GBRT': {
        'model': GradientBoostingRegressor,
        'params': [
            {'name': 'learning_rate', 'type': 'float', 'min': 0.01, 'max': 0.5, 'default': 0.1},
            {'name': 'n_estimators', 'type': 'int', 'min': 50, 'max': 500, 'default': 100},
            {'name': 'max_depth', 'type': 'int', 'min': 3, 'max': 10, 'default': 3}
        ]
    },
    'KRR': {
        'model': KernelRidge,
        'params': [
            {'name': 'alpha', 'type': 'float', 'min': 0.1, 'max': 10.0, 'default': 1.0},
            {'name': 'kernel', 'type': 'select', 'options': ['linear', 'rbf', 'poly'], 'default': 'rbf'}
        ]
    },
    'SVR': {
        'model': SVR,
        'params': [
            {'name': 'C', 'type': 'float', 'min': 0.1, 'max': 100.0, 'default': 1.0},
            {'name': 'epsilon', 'type': 'float', 'min': 0.01, 'max': 1.0, 'default': 0.1},
            {'name': 'kernel', 'type': 'select', 'options': ['linear', 'rbf', 'poly'], 'default': 'rbf'}
        ]
    }
}