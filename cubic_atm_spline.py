import numpy as np
from scipy.interpolate import CubicSpline


def estimate_atm_vol_cubic_spline(strikes, implied_vols, underlying_price):
    """
    Estimates the implied volatility at the 'true' ATM strike using cubic spline interpolation.

    Parameters
    ----------
    strikes : array-like
        Sorted array of strike prices for which implied_vols are known.
    implied_vols : array-like
        Implied volatilities corresponding to the strikes (must have same length).
    underlying_price : float
        The underlying's spot price (or forward price) considered the ATM strike.

    Returns
    -------
    float
        Interpolated (or extrapolated) implied volatility at the ATM strike.
    """
    # Ensure inputs are numpy arrays
    strikes = np.array(strikes, dtype=float)
    implied_vols = np.array(implied_vols, dtype=float)

    # Create the cubic spline interpolator
    spline = CubicSpline(strikes, implied_vols, bc_type='natural')

    # Evaluate the spline at the underlying price to get ATM implied vol
    atm_vol = spline(underlying_price)
    return atm_vol