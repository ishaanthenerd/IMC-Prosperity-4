"""
All products: alternate each tick between (1) a full buy ladder and (2) a full sell ladder inside
the spread—1 lot per price from bid+1 … ask−1. When |position| hits that product’s limit, flatten
with an aggressive order at best bid (long) or best ask (short). Uses the same stdout JSON Logger
as other algo traders.
"""

import json
from typing import Any

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)

# prosperity4bt `data.LIMITS` (extend if new products appear)
DEFAULT_POSITION_LIMIT = 80
PRODUCT_POSITION_LIMIT: dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}


def _position_limit(symbol: str) -> int:
    return PRODUCT_POSITION_LIMIT.get(symbol, DEFAULT_POSITION_LIMIT)


def _spread_prices(best_bid: int, best_ask: int) -> list[int]:
    """Every integer strictly between bid and ask."""
    if best_ask <= best_bid + 1:
        return []
    return list(range(best_bid + 1, best_ask))


class Logger:
    """Stdout JSON line for prosperity4bt / visualizer (same pattern as R1_davis)."""

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
                buyer = trade.buyer if trade.buyer is not None else ""
                seller = trade.seller if trade.seller is not None else ""
                compressed.append(
                    [
                        trade.symbol,
                        int(trade.price),
                        int(trade.quantity),
                        buyer,
                        seller,
                        int(trade.timestamp),
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
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    """Alternate buy-ladder / sell-ladder ticks; flatten at ±limit."""

    def __init__(self) -> None:
        # True ⇒ this tick uses buy-side ladder; False ⇒ sell-side ladder.
        self._buy_side_ladder_next = True

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        trader_data = ""
        conversions = 0

        use_buy_ladder = self._buy_side_ladder_next
        self._buy_side_ladder_next = not self._buy_side_ladder_next

        for symbol in state.order_depths:
            result[symbol] = self._orders_for_symbol(state, symbol, use_buy_ladder)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _orders_for_symbol(self, state: TradingState, symbol: str, use_buy_ladder: bool) -> list[Order]:
        depth = state.order_depths[symbol]
        pos = state.position.get(symbol, 0)
        lim = _position_limit(symbol)

        if not depth.buy_orders or not depth.sell_orders:
            return []

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # At limit: single aggressive order to flat (instasell / instabuy).
        if pos >= lim:
            return [Order(symbol, best_bid, -pos)]
        if pos <= -lim:
            return [Order(symbol, best_ask, -pos)]

        if best_ask <= best_bid + 1:
            return []

        prices = _spread_prices(best_bid, best_ask)
        if not prices:
            return []

        # Near ask first when clipping (same for buys and sells).
        prices_near_ask_first = sorted(prices, reverse=True)

        if use_buy_ladder:
            room = lim - pos
            if room <= 0:
                return []
            orders: list[Order] = []
            for p in prices_near_ask_first:
                if len(orders) >= room:
                    break
                if p <= 0:
                    continue
                orders.append(Order(symbol, int(p), 1))
            return orders

        # Sell-side ladder: 1 lot offered at each level (negative qty).
        room_sell = pos + lim
        if room_sell <= 0:
            return []
        orders = []
        for p in prices_near_ask_first:
            if len(orders) >= room_sell:
                break
            if p <= 0:
                continue
            orders.append(Order(symbol, int(p), -1))
        return orders
