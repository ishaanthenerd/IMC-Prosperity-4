import json
import math
import os
import statistics
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

FAIR_FALLBACK = 10_000
# Rolling mean of mid_price_using_best; window from env DAVIS_FAIR_MA_WINDOW (0 = spot mid only).
MM_BAND = 3
# Pepper (same as R1_davis_pepper): MA fair PEPPER_FAIR_MA_WINDOW; MM + optional ask-lift + MM ask cap.
# Signed inventory pct = 100*pos/LIMIT. MM_INVENTORY_THRESHOLD default 100*75/80 (sweep).
DEFAULT_MM_INVENTORY_THRESHOLD = 100.0 * 75 / 80
# Pepper: lift asks when inventory_pct < this (inventory percent). Env TAKING_THRESHOLD; none/off/disable = off.
DEFAULT_TAKING_THRESHOLD = 79.0
# PEPPER_MM_SELL_INV_CAP: cap MM asks so a full fill keeps inventory_pct > MM gate (default on).

# Position utilization → taking threshold (former "stricter" profile only).
# Utilization = |pos| / POSITION_LIMIT. Walk sorted keys; use last th with util >= key; else 0.
# Buy-take: ask < fair - th. Sell-take: bid > fair + th. Higher th = stricter; negative th = easier.
TAKING_THRESHOLD_BY_UTIL: dict[float, int] = {
    0.25: 4,
    0.5: 3,
    0.65: 2,
    0.8: 1,
    0.9: 0,
}


def _mm_inventory_threshold() -> float:
    raw = os.environ.get("MM_INVENTORY_THRESHOLD", "").strip()
    if not raw:
        return float(DEFAULT_MM_INVENTORY_THRESHOLD)
    return float(raw)


def _mm_sell_inventory_cap_enabled() -> bool:
    raw = os.environ.get("PEPPER_MM_SELL_INV_CAP", "1").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _taking_threshold_optional() -> float | None:
    """Pepper-only: env TAKING_THRESHOLD (same semantics as R1_davis_pepper)."""
    raw = os.environ.get("TAKING_THRESHOLD", "").strip()
    if raw == "":
        return float(DEFAULT_TAKING_THRESHOLD)
    if raw.lower() in ("none", "off", "disable"):
        return None
    return float(raw)


def _target_position_for_min_inventory_pct(min_pct: float) -> int:
    need = math.ceil(min_pct * POSITION_LIMIT / 100.0 - 1e-12)
    return max(-POSITION_LIMIT, min(POSITION_LIMIT, need))


def _min_position_for_inventory_pct_gt(th: float) -> int:
    lim = th * POSITION_LIMIT / 100.0
    return int(math.floor(lim + 1e-12)) + 1


def _max_mm_sell_respecting_inventory_floor(pos: int, mm_inv_threshold: float) -> int:
    if pos <= 0 or POSITION_LIMIT <= 0:
        return 0
    floor_pos = _min_position_for_inventory_pct_gt(mm_inv_threshold)
    return max(0, pos - floor_pos)


def _pepper_fair_ma_window() -> int:
    return int(os.environ.get("PEPPER_FAIR_MA_WINDOW", "10"))


def _taking_threshold_for_position(pos: int) -> int:
    util = abs(pos) / POSITION_LIMIT if POSITION_LIMIT else 0.0
    th = 0
    for frac in sorted(TAKING_THRESHOLD_BY_UTIL):
        if util >= frac:
            th = TAKING_THRESHOLD_BY_UTIL[frac]
    return th


def _record_abs_pos_sample(state: TradingState) -> None:
    path = os.environ.get("DAVIS_POS_STATS_FILE", "").strip()
    if not path:
        return
    pos = state.position.get(ASH_COATED_OSMIUM, 0)
    with open(path, "a", encoding="ascii") as f:
        f.write(f"{abs(pos)}\n")


def _mid_price_using_best(depth: OrderDepth) -> float:
    if not depth.buy_orders or not depth.sell_orders:
        return float("nan")
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _fair_values_and_trader_data(state: TradingState) -> tuple[int, int, str]:
    """Osmium fair (MA or spot), pepper fair (MA or spot), traderData JSON with mids and/or pepper_mids."""
    ash_window = int(os.environ.get("DAVIS_FAIR_MA_WINDOW", "10"))
    pepper_window = _pepper_fair_ma_window()
    td: dict[str, Any] = {}
    if state.traderData:
        try:
            td = json.loads(state.traderData)
        except json.JSONDecodeError:
            td = {}

    mids: list[float] = list(td.get("mids", []))
    ash_depth = state.order_depths.get(ASH_COATED_OSMIUM, None)
    if ash_depth is None:
        ash_fair = 0
    else:
        ash_mid = _mid_price_using_best(ash_depth)
        if ash_window <= 0:
            ash_fair = int(round(ash_mid)) if not math.isnan(ash_mid) else FAIR_FALLBACK
        else:
            if not math.isnan(ash_mid):
                mids.append(ash_mid)
                mids = mids[-ash_window:]
            ash_fair = int(round(statistics.mean(mids))) if mids else FAIR_FALLBACK

    pepper_mids: list[float] = list(td.get("pepper_mids", []))
    pepper_depth = state.order_depths[INTARIAN_PEPPER_ROOT]
    pepper_mid = _mid_price_using_best(pepper_depth)
    if pepper_window <= 0:
        pepper_fair = int(round(pepper_mid)) if not math.isnan(pepper_mid) else FAIR_FALLBACK
    else:
        if not math.isnan(pepper_mid):
            pepper_mids.append(pepper_mid)
            pepper_mids = pepper_mids[-pepper_window:]
        pepper_fair = int(round(statistics.mean(pepper_mids))) if pepper_mids else FAIR_FALLBACK

    out_td: dict[str, Any] = {}
    if ash_window > 0:
        out_td["mids"] = mids
    if pepper_window > 0:
        out_td["pepper_mids"] = pepper_mids

    return ash_fair, pepper_fair, json.dumps(out_td, separators=(",", ":"))


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
        _record_abs_pos_sample(state)
        result: dict[str, list[Order]] = {}
        conversions = 0
        ash_fair, pepper_fair, trader_data = _fair_values_and_trader_data(state)
        inv_th = _mm_inventory_threshold()
        pepper_take_th = _taking_threshold_optional()

        for product in state.order_depths:
            if product == INTARIAN_PEPPER_ROOT:
                result[product] = self._pepper_orders(state, pepper_fair, inv_th, pepper_take_th)
            elif product == ASH_COATED_OSMIUM:
                pass # result[product] = self._osmium_orders(state, ash_fair)
            else:
                result[product] = []

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _osmium_orders(self, state: TradingState, fair_value: int) -> list[Order]:
        depth = state.order_depths[ASH_COATED_OSMIUM]
        pos = state.position.get(ASH_COATED_OSMIUM, 0)
        orders: list[Order] = []

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        taking_th = _taking_threshold_for_position(pos)
        cheap_enough = best_ask is not None and best_ask < fair_value - taking_th
        rich_enough = best_bid is not None and best_bid > fair_value + taking_th

        take_orders: list[Order] = []
        sim = pos

        if cheap_enough and pos < 0:
            for price in sorted(depth.sell_orders.keys()):
                if price >= fair_value - taking_th:
                    continue
                vol = -depth.sell_orders[price]
                if vol <= 0:
                    continue
                room = min(POSITION_LIMIT - sim, -sim)
                if room <= 0:
                    break
                q = min(vol, room)
                if q <= 0:
                    continue
                take_orders.append(Order(ASH_COATED_OSMIUM, price, q))
                sim += q

        if rich_enough and pos > 0:
            for price in sorted(depth.buy_orders.keys(), reverse=True):
                if price <= fair_value + taking_th:
                    continue
                vol = depth.buy_orders[price]
                if vol <= 0:
                    continue
                room = min(sim + POSITION_LIMIT, sim)
                if room <= 0:
                    break
                q = min(vol, room)
                if q <= 0:
                    continue
                take_orders.append(Order(ASH_COATED_OSMIUM, price, -q))
                sim -= q

        pos_after_takes = sim

        lo, hi = fair_value - MM_BAND, fair_value + MM_BAND
        bids_outside = [p for p in depth.buy_orders if p < lo or p > hi]
        asks_outside = [p for p in depth.sell_orders if p < lo or p > hi]

        if bids_outside:
            mm_bid = max(bids_outside) + 1
        else:
            mm_bid = fair_value

        if asks_outside:
            mm_ask = min(asks_outside) - 1
        else:
            mm_ask = fair_value

        if mm_bid >= mm_ask:
            mm_ask = mm_bid + 1

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

    def _pepper_orders(
        self,
        state: TradingState,
        fair_value: int,
        mm_inv_threshold: float,
        taking_threshold: float | None,
    ) -> list[Order]:
        depth = state.order_depths[INTARIAN_PEPPER_ROOT]
        pos = state.position.get(INTARIAN_PEPPER_ROOT, 0)
        orders: list[Order] = []

        inventory_pct = 100.0 * pos / POSITION_LIMIT if POSITION_LIMIT else 0.0
        take_orders: list[Order] = []
        sim = pos

        if taking_threshold is not None and inventory_pct < taking_threshold:
            target_pos = _target_position_for_min_inventory_pct(taking_threshold)
            for price in sorted(depth.sell_orders.keys()):
                if sim >= target_pos:
                    break
                vol = -depth.sell_orders[price]
                if vol <= 0:
                    continue
                room = POSITION_LIMIT - sim
                need = target_pos - sim
                if room <= 0 or need <= 0:
                    break
                q = min(vol, room, need)
                if q <= 0:
                    continue
                take_orders.append(Order(INTARIAN_PEPPER_ROOT, price, q))
                sim += q

        pos_after_takes = sim
        inv_pct_after = 100.0 * pos_after_takes / POSITION_LIMIT if POSITION_LIMIT else 0.0
        post_asks = inv_pct_after > mm_inv_threshold

        lo, hi = fair_value - MM_BAND, fair_value + MM_BAND
        bids_outside = [p for p in depth.buy_orders if p < lo or p > hi]
        asks_outside = [p for p in depth.sell_orders if p < lo or p > hi]

        if bids_outside:
            mm_bid = max(bids_outside) + 1
        else:
            mm_bid = fair_value

        if asks_outside:
            mm_ask = min(asks_outside) - 1
        else:
            mm_ask = fair_value

        if post_asks and mm_bid >= mm_ask:
            mm_ask = mm_bid + 1

        mm_buy_sz = max(0, POSITION_LIMIT - pos_after_takes)
        mm_sell_sz = max(0, pos_after_takes + POSITION_LIMIT)
        if post_asks and _mm_sell_inventory_cap_enabled():
            inv_cap = _max_mm_sell_respecting_inventory_floor(pos_after_takes, mm_inv_threshold)
            mm_sell_sz = min(mm_sell_sz, inv_cap)

        mm_orders: list[Order] = []
        if mm_buy_sz > 0:
            mm_orders.append(Order(INTARIAN_PEPPER_ROOT, mm_bid, mm_buy_sz))
        if post_asks and mm_sell_sz > 0:
            mm_orders.append(Order(INTARIAN_PEPPER_ROOT, mm_ask, -mm_sell_sz))

        orders.extend(take_orders)
        orders.extend(mm_orders)
        return orders
