# File: backend/app/services/preprocessors.py


# Note: The implementation of apply_continuum_removal is complex.
# This implementation uses NumPy to find the convex hull and interpolate
# the "upper envelope" for the removal process.

#### apply_reflectance :
"""
Normalizes or clips spectral values to the [0, 1] range.
[cite_start][cite: 101]

Args:
    df_spectral (pd.DataFrame): DataFrame of spectral data.

Returns:
    pd.DataFrame: DataFrame with values clipped to [0, 1].
"""
- Use numpy.clip to efficiently bound all values

#### apply_absorbance : 
"""
Computes absorbance from reflectance: -log(reflectance + eps).
[cite_start][cite: 101]

Args:
    df_spectral (pd.DataFrame): DataFrame of spectral data.
    eps (float, optional): Epsilon value for numerical stability 
                            to avoid log(0). [cite_start]Defaults to 1e-8. [cite: 101]

Returns:
    pd.DataFrame: DataFrame of computed absorbance values.
"""
- Apply the absorbance transformation safely using the epsilon

#### apply_continuum_removal :
"""
Applies continuum removal to each spectrum (row) in the DataFrame.

This method finds the "upper envelope" (convex hull) of the spectrum
[cite_start]and divides the original spectrum by this envelope. [cite: 105]

Args:
    df_spectral (pd.DataFrame): DataFrame of spectral data, where
                                    each row is one spectrum.

Returns:
    pd.DataFrame: DataFrame with continuum-removed spectra.
"""