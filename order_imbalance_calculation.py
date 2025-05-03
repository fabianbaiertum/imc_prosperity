import numpy as np
def exponential_order_imbalance(bid_volumes, ask_volumes, alpha_decay):
    """
    Computes a symmetric order imbalance measure for the first 3 levels
    of the book, giving higher weight to volumes at better (closer) levels
    via an exponential decay.

    The returned imbalance I is in [-1, +1], where:

I = +1 means all weighted volume is on the bid side.
I = -1 means all weighted volume is on the ask side.
I =  0 means balanced weighted volumes.

    Parameters
    ----------
    bid_volumes : list or array-like of length 3
        bid_volumes[i] is the volume at level i (i=0 is best, i=1 is 2nd, i=2 is 3rd).
    ask_volumes : list or array-like of length 3
        ask_volumes[i] is the volume at level i (i=0 is best, i=1 is 2nd, i=2 is 3rd).
    alpha_decay : float
        Decay parameter for exponential weighting. Higher means
        we discount deeper levels more sharply.

    Returns
    -------
    float
        The symmetric order imbalance in [-1, +1].
    """

    # Convert inputs to NumPy arrays
    bid_volumes = np.array(bid_volumes, dtype=float)
    ask_volumes = np.array(ask_volumes, dtype=float)

    # Level indices: 0=best, 1=2nd best, 2=3rd best
    levels = np.arange(len(bid_volumes))

    # Exponential weights: w(i) = exp(-alpha_decay * i)
    weights = np.exp(-alpha_decay * levels)

    # Weighted sums
    weighted_bid = np.sum(bid_volumes * weights)
    weighted_ask = np.sum(ask_volumes * weights)

    denominator = weighted_bid + weighted_ask
    if denominator == 0:
        # Edge case: no volume at any level
        # Return 0 to indicate perfectly balanced (or effectively no data)
        return 0.0

    # Symmetric imbalance: in [-1, +1]
    imbalance = (weighted_bid - weighted_ask) / denominator

    return imbalance