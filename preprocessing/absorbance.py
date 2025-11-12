import pandas as pd
import numpy as np

def apply_absorbance(df_spectral: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """
    Computes absorbance from reflectance: -log(reflectance + eps).
    """
    df_absorbance = -np.log(df_spectral + eps)
    return df_absorbance