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

INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"
ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMIT = 80

FAIR_VALUE = 10_000
# MM: undercut best bid / best ask outside [FAIR - band, FAIR + band]; else anchor at fair (+ spread fix).
MM_BAND = 3
TAKING_THRESHOLD = 3


class Logger:
    """Same stdout format as R1_davis_old so backtest `lambdaLog` is populated."""

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
        """Rows [symbol, price, qty, buyer, seller, ts] — jmerle P3 visualizer splits own vs bot by own_trades vs market_trades arrays."""
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
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        trader_data = ""
        conversions = 0

        for product in state.order_depths:
            if product == INTARIAN_PEPPER_ROOT:
                result[product] = []
            elif product == ASH_COATED_OSMIUM:
                result[product] = self._osmium_orders(state)
            else:
                result[product] = []

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _osmium_orders(self, state: TradingState) -> list[Order]:
        depth = state.order_depths[ASH_COATED_OSMIUM]
        pos = state.position.get(ASH_COATED_OSMIUM, 0)
        orders: list[Order] = []

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        cheap_enough = best_ask is not None and best_ask < FAIR_VALUE - TAKING_THRESHOLD
        rich_enough = best_bid is not None and best_bid > FAIR_VALUE + TAKING_THRESHOLD

        # TAKING SECTION — consumes position headroom first; MM only sees what is left.
        take_orders: list[Order] = []
        sim = pos

        if cheap_enough and pos < 0:
            for price in sorted(depth.sell_orders.keys()):
                if price >= FAIR_VALUE - TAKING_THRESHOLD:
                    continue
                vol = -depth.sell_orders[price]
                if vol <= 0:
                    continue
                room = POSITION_LIMIT - sim
                if room <= 0:
                    break
                q = min(vol, room)
                if q <= 0:
                    continue
                take_orders.append(Order(ASH_COATED_OSMIUM, price, q))
                sim += q

        if rich_enough and pos > 0:
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price <= FAIR_VALUE + TAKING_THRESHOLD:
                    continue
                vol = depth.buy_orders[price]
                if vol <= 0:
                    continue
                room = sim + POSITION_LIMIT
                if room <= 0:
                    break
                q = min(vol, room)
                if q <= 0:
                    continue
                take_orders.append(Order(ASH_COATED_OSMIUM, price, -q))
                sim -= q

        pos_after_takes = sim

        # MM SECTION — sized from position after takes so volume is left only if takes did not use it.
        lo, hi = FAIR_VALUE - MM_BAND, FAIR_VALUE + MM_BAND
        bids_outside = [p for p in depth.buy_orders if p < lo or p > hi]
        asks_outside = [p for p in depth.sell_orders if p < lo or p > hi]

        if bids_outside:
            mm_bid = max(bids_outside) + 1
        else:
            mm_bid = FAIR_VALUE

        if asks_outside:
            mm_ask = min(asks_outside) - 1
        else:
            mm_ask = FAIR_VALUE

        if mm_bid >= mm_ask:
            mm_ask = mm_bid + 1

        # If every buy and every sell in this batch fills: pos_after_takes + B_mm - S_mm in band.
        # B_mm = LIMIT - pos_after_takes, S_mm = pos_after_takes + LIMIT => joint fill at -pos_after_takes.
        mm_buy_sz = max(0, POSITION_LIMIT - pos_after_takes)
        mm_sell_sz = max(0, pos_after_takes + POSITION_LIMIT)

        mm_orders: list[Order] = []
        if mm_buy_sz > 0:
            mm_orders.append(Order(ASH_COATED_OSMIUM, mm_bid, mm_buy_sz))
        if mm_sell_sz > 0:
            mm_orders.append(Order(ASH_COATED_OSMIUM, mm_ask, -mm_sell_sz))

        orders.extend(take_orders)
        orders.extend(mm_orders)
        return orders
