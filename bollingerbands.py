from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState, UserId
)
from typing import List, Any, Tuple, Dict
import jsonpickle
import math
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return jsonpickle.encode(value, unpicklable=False)

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()

class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

##############################################################################
# Helper functions for Bollinger Weighted Moving Average
##############################################################################

def bollinger_moving_average(prices: List[float], window: int) -> float:
    """
    Computes a Weighted Moving Average of the last `window` prices using
    descending weights (window, window-1, ..., 1).
    """
    recent_prices = prices[-window:]
    weights = np.arange(window, 0, -1)  # e.g. for window=5 => [5,4,3,2,1]
    norm = np.sum(weights)
    return np.dot(recent_prices, weights) / norm

def bmwa_std(prices: List[float], window: int) -> float:
    """
    Computes the sample standard deviation of the last `window` prices around
    the Bollinger Weighted Moving Average.
    """
    recent_prices = prices[-window:]
    wma = bollinger_moving_average(prices, window)
    squared_devs = (recent_prices - wma) ** 2
    # Use sample std with (window - 1) in the denominator
    return np.sqrt(np.sum(squared_devs) / (window - 1))

##############################################################################
# Default strategy parameters
##############################################################################

PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    Product.KELP: {
        "take_width": 1,
        "position_limit": 50,
        "min_volume_filter": 20,
        "spread_edge": 1,
        "default_fair_method": "vwap_with_vol_filter",
    },
    Product.SQUID_INK: {
        # Bollinger Weighted MA parameters:
        "bmwa_window": 2,        # how many recent mid-prices to weight
        "bmwa_k": 0.02,            # how many std devs away to form upper/lower band (has to be a small number, since we don't divide by the price in the std calculation
        # More general SQUID_INK params:
        "spread_edge": 100,
        "position_limit": 50,
        "take_width": 0.3,
    },
}

##############################################################################
# Trader class
##############################################################################

class Trader:
    def __init__(self, params: Dict[str, Any] = None) -> None:
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50
        }

        self.kelp_prices: List[float] = []
        self.kelp_vwap: List[float] = []
        # Stores all mid-prices we compute for SQUID_INK
        self.ink_prices: List[float] = []

    # -----------------------------------------------------------------------
    # RESIN Strategy (unchanged)
    # -----------------------------------------------------------------------

    def resin_take_orders(self, order_depth: OrderDepth, fair_value: float, position: int, position_limit: int
                          ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(Product.RESIN, best_bid, -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def resin_clear_orders(self, order_depth: OrderDepth, position: int, fair_value: float, position_limit: int,
                           buy_order_volume: int, sell_order_volume: int
                           ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(Product.RESIN, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(Product.RESIN, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def resin_make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, position_limit: int,
                          buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        orders: List[Order] = []
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf_val = max(bbbf) if bbbf else fair_value - 2

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.RESIN, bbbf_val + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.RESIN, baaf - 1, -sell_quantity))

        return orders

    # -----------------------------------------------------------------------
    # KELP Strategy (unchanged)
    # -----------------------------------------------------------------------

    def kelp_fair_value(self, order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2

        elif method == "mid_price_with_vol_filter":
            sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
            buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
            if not sell_orders or not buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
            else:
                best_ask = min(sell_orders)
                best_bid = max(buy_orders)
            return (best_ask + best_bid) / 2

        elif method == "vwap":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume

        elif method == "vwap_with_vol_filter":
            sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
            buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
            if not sell_orders or not buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
                if volume == 0:
                    return (best_ask + best_bid) / 2
                return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
            else:
                best_ask = min(sell_orders)
                best_bid = max(buy_orders)
                volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
                if volume == 0:
                    return (best_ask + best_bid) / 2
                return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
        else:
            raise ValueError("Unknown fair value method specified.")

    def kelp_take_orders(self, order_depth: OrderDepth, fair_value: float, params: dict, position: int
                         ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
                quantity = min(ask_amount, params["position_limit"] - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
                quantity = min(bid_amount, params["position_limit"] + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def kelp_clear_orders(self, order_depth: OrderDepth, position: int, params: dict, fair_value: float,
                          buy_order_volume: int, sell_order_volume: int
                          ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = params["position_limit"] - (position + buy_order_volume)
        sell_quantity = params["position_limit"] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order("KELP", fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order("KELP", fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def kelp_make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, params: dict,
                         buy_order_volume: int, sell_order_volume: int
                         ) -> List[Order]:
        orders: List[Order] = []
        edge = params["spread_edge"]
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
        baaf = min(aaf) if aaf else fair_value + edge + 1
        bbbf = max(bbf) if bbf else fair_value - edge - 1

        buy_quantity = params["position_limit"] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))

        sell_quantity = params["position_limit"] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders

    # -----------------------------------------------------------------------
    # SQUID_INK Strategy with Bollinger Weighted Moving Average
    # -----------------------------------------------------------------------

    def ink_take_orders(self, order_depth: OrderDepth, fair_value: float, params: dict, position: int
                        ) -> Tuple[List[Order], int, int]:
        """
        Evaluate buy/sell signals based on whether the latest mid-price
        is above/below the BWMA ± K·std. If price > upper band => Sell.
        If price < lower band => Buy.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Track the fair (mid) price for SQUID_INK
        self.ink_prices.append(fair_value)

        window = params["bmwa_window"]
        k = params["bmwa_k"]

        # Only proceed if we have enough data
        if len(self.ink_prices) >= window:
            bwma = bollinger_moving_average(self.ink_prices, window)
            std_dev = bmwa_std(self.ink_prices, window)

            upper_band = bwma + k * std_dev
            lower_band = bwma - k * std_dev

            # ----------------------------------------------
            # SELL signal: if latest price > upper band
            # ----------------------------------------------
            if self.ink_prices[-1] > upper_band:
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    bid_amount = order_depth.buy_orders[best_bid]
                    # We can be aggressive if the best_bid is high enough
                    if best_bid >= fair_value + params["take_width"]:
                        quantity = min(bid_amount, params["position_limit"] + position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID_INK, best_bid, -quantity))
                            sell_order_volume += quantity

            # ----------------------------------------------
            # BUY signal: if latest price < lower band
            # ----------------------------------------------
            elif self.ink_prices[-1] < lower_band:
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    ask_amount = -order_depth.sell_orders[best_ask]
                    if best_ask <= fair_value - params["take_width"]:
                        quantity = min(ask_amount, params["position_limit"] - position)
                        if quantity > 0:
                            orders.append(Order(Product.SQUID_INK, best_ask, quantity))
                            buy_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def ink_clear_orders(self, order_depth: OrderDepth, position: int, params: dict, fair_value: float,
                         buy_order_volume: int, sell_order_volume: int
                         ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = params["position_limit"] - (position + buy_order_volume)
        sell_quantity = params["position_limit"] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(Product.SQUID_INK, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(Product.SQUID_INK, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return orders, buy_order_volume, sell_order_volume

    def ink_make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, params: dict,
                        buy_order_volume: int, sell_order_volume: int
                        ) -> List[Order]:
        """
        Place passive buy/sell orders away from current mid-price by spread_edge,
        to catch trades if the market moves our way.
        """
        orders: List[Order] = []
        edge = params["spread_edge"]
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
        baaf = min(aaf) if aaf else int(round(fair_value + edge + 1))
        bbbf = max(bbf) if bbf else int(round(fair_value - edge - 1))

        buy_quantity = params["position_limit"] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.SQUID_INK, bbbf + 1, buy_quantity))

        sell_quantity = params["position_limit"] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.SQUID_INK, baaf - 1, -sell_quantity))

        return orders

    # -----------------------------------------------------------------------
    # Main run() function
    # -----------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        # ------------------ Process RESIN ------------------
        if Product.RESIN in self.params and Product.RESIN in state.order_depths:
            resin_position = state.position.get(Product.RESIN, 0)
            resin_params = self.params[Product.RESIN]
            resin_order_depth = state.order_depths[Product.RESIN]
            resin_fair_value = resin_params["fair_value"]  # For RESIN, fair value is fixed.

            orders_take, buy_vol, sell_vol = self.resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RESIN]
            )
            orders_clear, buy_vol, sell_vol = self.resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, self.LIMIT[Product.RESIN],
                buy_vol, sell_vol
            )
            orders_make = self.resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RESIN],
                buy_vol, sell_vol
            )
            result[Product.RESIN] = orders_take + orders_clear + orders_make

        # ------------------ Process KELP ------------------
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_params = self.params[Product.KELP]
            kelp_order_depth = state.order_depths[Product.KELP]
            kelp_fair_value = self.kelp_fair_value(
                kelp_order_depth,
                kelp_params["default_fair_method"],
                kelp_params["min_volume_filter"]
            )
            kelp_take_orders, buy_vol, sell_vol = self.kelp_take_orders(
                kelp_order_depth, kelp_fair_value, kelp_params, kelp_position
            )
            kelp_clear_orders, buy_vol, sell_vol = self.kelp_clear_orders(
                kelp_order_depth, kelp_position, kelp_params, kelp_fair_value, buy_vol, sell_vol
            )
            kelp_make_orders = self.kelp_make_orders(
                kelp_order_depth, kelp_fair_value, kelp_position, kelp_params, buy_vol, sell_vol
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders

        # ------------------ Process SQUID_INK (BWMA) ------------------
        if Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_params = self.params[Product.SQUID_INK]
            ink_order_depth = state.order_depths[Product.SQUID_INK]

            # Simple mid-price as a baseline 'fair_value'
            best_ask = min(ink_order_depth.sell_orders.keys())
            best_bid = max(ink_order_depth.buy_orders.keys())
            ink_fair_value = (best_ask + best_bid) // 2

            ink_take_orders, buy_vol, sell_vol = self.ink_take_orders(
                ink_order_depth, ink_fair_value, ink_params, ink_position
            )
            ink_clear_orders, buy_vol, sell_vol = self.ink_clear_orders(
                ink_order_depth, ink_position, ink_params, ink_fair_value, buy_vol, sell_vol
            )
            ink_make_orders = self.ink_make_orders(
                ink_order_depth, ink_fair_value, ink_position, ink_params, buy_vol, sell_vol
            )
            result[Product.SQUID_INK] = ink_take_orders + ink_clear_orders + ink_make_orders

        # Convert any leftover stuff, log, etc.
        traderData = jsonpickle.encode(
            {
                "kelp_prices": self.kelp_prices,
                "kelp_vwap": self.kelp_vwap,
                "ink_prices": self.ink_prices,
            },
            unpicklable=False
        )
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
