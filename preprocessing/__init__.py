"""
Preprocessing package for Spectral Soil Modeler.
Contains Reflectance, Absorbance, and Continuum Removal transformations.
"""
from .reflectance import apply_reflectance
from .absorbance import apply_absorbance
from .continuum_removal import apply_continuum_removal

PREPROCESSING_METHODS = {
    "Reflectance": apply_reflectance,
    "Absorbance": apply_absorbance,
    "ContinuumRemoval": apply_continuum_removal
}