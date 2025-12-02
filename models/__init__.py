"""
Models package for Spectral Soil Modeler.
Contains definitions for all regression algorithms.
"""
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from cubist import Cubist

MODEL_CONFIG = {
    'PLSR': {
        'model': PLSRegression,
        'params': [
            {'name': 'n_components', 'type': 'int', 'values': list(range(1, 51)), 'default': 10}
        ]
    },
    'Cubist': {
        'model': Cubist,
        'params': [
            {'name': 'n_rules', 'type': 'int', 'values': list(range(10, 501, 10)), 'default': 100}
        ]
    },
    'GBRT': {
        'model': GradientBoostingRegressor,
        'params': [
            {'name': 'learning_rate', 'type': 'float', 'values': [0.01, 0.05, 0.1, 0.2, 0.3], 'default': 0.1},
            {'name': 'n_estimators', 'type': 'int', 'values': list(range(50, 501, 50)), 'default': 100},
            {'name': 'max_depth', 'type': 'int', 'values': list(range(2, 11)), 'default': 3}
        ]
    },
    'KRR': {
        'model': KernelRidge,
        'params': [
            {'name': 'alpha', 'type': 'float', 'values': [0.1, 0.5, 1.0, 2.0, 5.0], 'default': 1.0},
            {'name': 'kernel', 'type': 'select', 'values': ['linear', 'rbf', 'poly'], 'default': 'rbf'}
        ]
    },
    'SVR': {
        'model': SVR,
        'params': [
            {'name': 'C', 'type': 'float', 'values': [0.1, 1, 10, 50, 100], 'default': 1.0},
            {'name': 'epsilon', 'type': 'float', 'values': [0.01, 0.05, 0.1, 0.2, 0.5], 'default': 0.1},
            {'name': 'kernel', 'type': 'select', 'values': ['linear', 'rbf', 'poly'], 'default': 'rbf'}
        ]
    }
}
