import numpy as np

def arima_forecast(series, p=1, d=0, q=0):
    """
    Simple ARIMA(p,d,q) forecast using NumPy.
    Only outputs 1-step ahead forecast.

    Parameters
    ----------
    series : list or np.array
        Original time series data (latest value last). Here we input our array of previously estimated theoretical prices, e.g. with my microprice calculation.
    p : int
        Number of AR (autoregressive) terms.
    d : int
        Number of differencing operations.
    q : int
        Number of MA (moving average) terms.

    Returns
    -------
    forecast : float
        Forecasted next value in the time series.
    """
    series = np.asarray(series, dtype=float)
    if len(series) < max(p, q) + d + 1:
        raise ValueError("Series too short for given p, d, q")

    # Step 1: Differencing to make series stationary
    diff_series = series.copy()
    for _ in range(d):
        diff_series = np.diff(diff_series)

    # Step 2: Fit AR(p) using least squares
    if p > 0:
        X_ar = np.array([diff_series[-p - i: -i] for i in range(1, q + 2)]).squeeze()
        if q == 0:
            X_ar = diff_series[-p:]
        y_ar = diff_series[-p:]
        ar_coeffs = np.linalg.lstsq(X_ar.reshape(-1, p), y_ar, rcond=None)[0]
    else:
        ar_coeffs = np.array([])

    # Step 3: Fit MA(q) by estimating past residuals (simplified)
    residuals = []
    if q > 0:
        for i in range(q):
            if p > 0:
                pred = np.dot(diff_series[-p - i: -i], ar_coeffs)
            else:
                pred = 0
            residuals.append(diff_series[-i - 1] - pred)
        residuals = np.array(residuals)
        ma_coeffs = np.linalg.lstsq(np.eye(q), residuals, rcond=None)[0]
    else:
        ma_coeffs = np.array([])

    # Step 4: Predict next differenced value
    forecast_diff = 0
    if p > 0:
        forecast_diff += np.dot(ar_coeffs, diff_series[-p:])
    if q > 0:
        forecast_diff += np.dot(ma_coeffs, residuals)

    # Step 5: Invert differencing to get back to original scale
    forecast = forecast_diff
    last_value = series[-1]
    for _ in range(d):
        forecast += last_value
        last_value = forecast

    return forecast
