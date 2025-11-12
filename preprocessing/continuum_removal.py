import pandas as pd
import numpy as np

def apply_continuum_removal(df_spectral: pd.DataFrame) -> pd.DataFrame:
    """
    Applies continuum removal to each spectrum (row) in the DataFrame
    using a NumPy-only upper convex hull algorithm.
    """
    wavelengths = np.arange(df_spectral.shape[1])

    df_continuum_removed = pd.DataFrame(index=df_spectral.index, columns=df_spectral.columns, dtype=np.float64)

    for i, (index, spectrum) in enumerate(df_spectral.iterrows()):
        spectrum_values = spectrum.values
        points = np.column_stack((wavelengths, spectrum_values))

        current_hull = [0] 
        for k in range(1, len(wavelengths)):
            current_hull.append(k)
            
            while len(current_hull) >= 3:
                p1_idx = current_hull[-3]
                p2_idx = current_hull[-2]
                p3_idx = current_hull[-1]
                
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                p3 = points[p3_idx]
                
                cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                
                if cross_product >= 0: 
                    break
                else:
                    current_hull.pop(-2)
                    
        hull_wavelengths = wavelengths[current_hull]
        hull_values = spectrum_values[current_hull]
        
        continuum_envelope = np.interp(wavelengths, hull_wavelengths, hull_values)

        removed_spectrum = np.where(continuum_envelope > 0, spectrum_values / continuum_envelope, spectrum_values)
        
        df_continuum_removed.iloc[i] = removed_spectrum

    return df_continuum_removed