import numpy as np

def rsi_vol_position_size(rsi_value, volatility,
                          alpha_rsi=0.1,
                          vol_scale_factor=1.0,
                          max_position=1.0):
    """
    Computes a position size based on an RSI oscillator and volatility.
    Parameters
    ----------
    rsi_value : float
        The current RSI reading (0 - 100).
    volatility : float
        A measure of recent volatility (e.g. std dev of returns, or ATR).
        Must be positive. Higher = more volatile.
    alpha_rsi : float
        Exponential factor for RSI-based scaling.
        Larger => sharper exponential growth for extreme RSI values.
    vol_scale_factor : float
        A constant scaling that transforms volatility into a position-size
        multiplier. For example, set vol_scale_factor=1.0 and you'll
        get a 1/volatility-type scaling. Adjust to taste.
    max_position : float
        An absolute cap on the final position size (in units of your choice).
        For instance, 1.0 might represent the maximum number of contracts
        you want to hold, or a fraction of capital, etc.
    Returns
    -------
    float
        The final position size, positive for long, negative for short,
        or 0 if no trade signal.
        The magnitude is capped by `max_position`.
    """
    # ------------------------
    # 1) Volatility Adjustment
    #    e.g., we do: vol_adj = vol_scale_factor / volatility
    #    so if volatility is large, vol_adj gets smaller => smaller position
    # ------------------------
    if volatility <= 0:
        # Avoid divide-by-zero. If volatility is zero or negative (edge case),
        # treat as minimal or no volatility => largest size
        vol_adj = vol_scale_factor
    else:
        vol_adj = vol_scale_factor / volatility

    # We can clamp it so we don't get extremely large positions
    # in case of extremely low volatility:
    vol_adj = min(vol_adj, 1.0)  # you can adjust this clamp as needed

    # ------------------------
    # 2) RSI-Sensitive Scaling
    #    We'll define two exponential curves:
    #      - If RSI>70 => short side
    #      - If RSI<30 => long side
    #    We want the exponent to increase as RSI moves toward the extremes:
    #      - RSI from 70 to 99 => scale from near 0 up to a high factor
    #      - RSI from 30 down to 1 => similarly for the long side
    #
    #    We'll make the function "start" near 0 at boundary 70 or 30
    #    and grow exponentially as RSI moves away from that boundary.
    # ------------------------

    # Default: no position
    rsi_factor = 0.0
    position_side = 0  # +1 => long, -1 => short, 0 => none

    if rsi_value > 70:
        # Example exponential: scale from RSI=70 to RSI=99
        # distance = (rsi_value - 70), so if RSI=70 => distance=0 => factor=~0
        # if RSI=90 => distance=20 => factor grows
        distance = rsi_value - 70
        # exponential growth
        # offsetting by -1 so that factor=0 at RSI=70
        rsi_factor = np.exp(alpha_rsi * distance) - 1.0
        position_side = -1  # short

    elif rsi_value < 30:
        # from RSI=30 to RSI=1
        distance = 30 - rsi_value
        # similarly, factor=0 at RSI=30, grows as RSI -> 1
        rsi_factor = np.exp(alpha_rsi * distance) - 1.0
        position_side = +1  # long

    # If 30 <= RSI <= 70, we do not trade => rsi_factor=0 => position=0

    # Ensure factor is non-negative
    rsi_factor = max(rsi_factor, 0.0)

    # ------------------------
    # 3) Combine the two scalars
    #      final_size = sign * (volatility_sizing * rsi_sizing)
    # ------------------------
    raw_size = position_side * (vol_adj * rsi_factor)

    # Cap the absolute size to max_position
    final_size = np.sign(raw_size) * min(abs(raw_size), max_position)

    return final_size