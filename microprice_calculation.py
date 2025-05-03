import numpy as np
def distance_weighted_microprice(bid_prices, bid_volumes,
                                 ask_prices, ask_volumes,
                                 alpha_decay):
    """
    Computes a distance-based microprice with exponential decay weighting.

    Parameters
    ----------
    bid_prices : list or 1D array of floats
        Bid price levels (e.g., [100.00, 99.90, 99.80, ...])
    bid_volumes : list or 1D array of floats
        Corresponding volumes at each bid price
    ask_prices : list or 1D array of floats
        Ask price levels (e.g., [100.10, 100.20, 100.30, ...])
    ask_volumes : list or 1D array of floats
        Corresponding volumes at each ask price
    alpha_decay : float
        Positive decay parameter for the exponential weighting
        (larger => more aggressive decay)

    Returns
    -------
    float
        The distance-weighted microprice.
    """

    bid_prices = np.array(bid_prices, dtype=float)
    bid_volumes = np.array(bid_volumes, dtype=float)
    ask_prices = np.array(ask_prices, dtype=float)
    ask_volumes = np.array(ask_volumes, dtype=float)

    # 1) Identify best bid and best ask for the mid price
    #    (Assuming your first elements are the best bid/ask)
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    #mid_price = (best_bid + best_ask) / 2.0
    best_microprice= (best_bid*ask_volumes[0] + best_ask*bid_volumes[0])/(ask_volumes[0]+bid_volumes[0])

    # 2) Compute distances from mid price
    bid_distances = np.abs(bid_prices - best_microprice)
    ask_distances = np.abs(ask_prices - best_microprice)

    # 3) Compute exponential weights: w(d) = exp(-alpha_decay * d)
    bid_weights = np.exp(-alpha_decay * bid_distances)
    ask_weights = np.exp(-alpha_decay * ask_distances)

    # 4) Compute weighted notional and weighted volume for each side
    weighted_notional_bid = np.sum(bid_prices * bid_volumes * bid_weights)
    weighted_volume_bid = np.sum(bid_volumes * bid_weights)

    weighted_notional_ask = np.sum(ask_prices * ask_volumes * ask_weights)
    weighted_volume_ask = np.sum(ask_volumes * ask_weights)

    # 5) Combine to get the microprice
    total_weighted_notional = weighted_notional_bid + weighted_notional_ask
    total_weighted_volume = weighted_volume_bid + weighted_volume_ask

    # Safety check to avoid divide-by-zero
    if total_weighted_volume == 0:
        return best_microprice
        #return mid_price


    microprice = total_weighted_notional / total_weighted_volume
    return microprice