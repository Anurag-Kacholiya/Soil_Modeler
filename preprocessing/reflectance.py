import pandas as pd
import numpy as np

def apply_reflectance(df_spectral: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes or clips spectral values to the [0, 1] range.
    """
    df_clipped = df_spectral.clip(lower=0.0, upper=1.0)
    return df_clipped