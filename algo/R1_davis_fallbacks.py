"""
Round 1: osmium logic copied from R1_davis_final; pepper uses L2-based fair, delta/trend/reversal
assumptions, and the take / initial_buy / market_make pipeline described in-module.
"""

import json
import math
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

# --- shared ---
POSITION_LIMIT = 80
FAIR_FALLBACK = 10_000
MM_BAND = 3

# === PEPPER ===
INTARIAN_PEPPER_ROOT = "INTARIAN_PEPPER_ROOT"
INITIAL_BUY_INVENTORY_TARGET = 75
# initial_buy: ladder mid+1..mid+4 with weights (21,18,15,12); total scaled by cap while preserving ratios.
INITIAL_BUY_ORDER_QTY = 75
INITIAL_BUY_LADDER_OFFSETS = (1, 2, 3, 4)  # vs int(round(L1 mid)); e.g. mid 11998 → 11999..12002
INITIAL_BUY_LADDER_WEIGHTS = (21, 18, 15, 12)  # sum 66; _alloc_by_weights keeps ratios under limits
MM_SELL_VOL_HIGH = 7
MM_SELL_VOL_MID = 5
MM_SELL_VOL_LOW = 2
FAIR_DROP_TREND_OFF = 5
MID_UP_STREAK_ON = 20  # trend_assumption turns True after this many consecutive mid[t] >= mid[t-1]
REVERSAL_SAMPLE_MIN = 12  # need more than this many large-delta samples
REVERSAL_LARGE_ABS = 3
REVERSAL_BAD_FRAC = 0.5  # more than half didn't reverse -> False
# Min inventory after profitable sell-takes when trend_assumption is True (caps hit size).
# Set to -POSITION_LIMIT to only use the usual position limit.
DISABLE_PROFITABLE_SELL_TAKE = 80
# When trend_assumption is True, suppress MM ask size until pos reaches this level. When trend is False, MM sells stay on.
DISABLE_MARKET_MAKE_SELLS = 70
MM_BAND_PEPPER = 5

# === ROOTS (osmium) ===
ASH_COATED_OSMIUM = "ASH_COATED_OSMIUM"
DAVIS_FAIR_MA_WINDOW = 10
DAVIS_POS_STATS_FILE = ""
TAKING_THRESHOLD_BY_UTIL: dict[float, int] = {
    0.25: 4,
    0.5: 3,
    0.65: 2,
    0.8: 1,
    0.9: 0,
}


def _taking_threshold_for_position(pos: int) -> int:
    util = abs(pos) / POSITION_LIMIT if POSITION_LIMIT else 0.0
    th = 0
    for frac in sorted(TAKING_THRESHOLD_BY_UTIL):
        if util >= frac:
            th = TAKING_THRESHOLD_BY_UTIL[frac]
    return th


def _record_abs_pos_sample(state: TradingState) -> None:
    if not DAVIS_POS_STATS_FILE:
        return
    pos = state.position.get(ASH_COATED_OSMIUM, 0)
    with open(DAVIS_POS_STATS_FILE, "a", encoding="ascii") as f:
        f.write(f"{abs(pos)}\n")


def _mid_price_using_best(depth: OrderDepth) -> float:
    if not depth.buy_orders or not depth.sell_orders:
        return float("nan")
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _l2_bid_ask(depth: OrderDepth) -> tuple[float, float] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bids = sorted(depth.buy_orders.keys(), reverse=True)
    asks = sorted(depth.sell_orders.keys())
    if len(bids) < 2 or len(asks) < 2:
        return None
    return float(bids[1]), float(asks[1])


def fair_mid(depth: OrderDepth) -> float:
    """Pepper fair anchor: mean(L2 bid, L2 ask) when L2 exists, else L1 mid."""
    if not depth.buy_orders or not depth.sell_orders:
        return float("nan")
    l2 = _l2_bid_ask(depth)
    if l2 is not None:
        l2b, l2a = l2
        return (l2b + l2a) / 2.0
    return _mid_price_using_best(depth)


def _ash_fair_and_mids(state: TradingState, td: dict[str, Any]) -> tuple[int, list[float]]:
    mids: list[float] = list(td.get("mids", []))
    ash_depth = state.order_depths.get(ASH_COATED_OSMIUM, None)
    if ash_depth is None:
        return 0, mids
    ash_mid = _mid_price_using_best(ash_depth)
    if DAVIS_FAIR_MA_WINDOW <= 0:
        ash_fair = int(round(ash_mid)) if not math.isnan(ash_mid) else FAIR_FALLBACK
    else:
        if not math.isnan(ash_mid):
            mids.append(ash_mid)
            mids = mids[-DAVIS_FAIR_MA_WINDOW :]
        ash_fair = int(round(statistics.mean(mids))) if mids else FAIR_FALLBACK
    return ash_fair, mids


def calc_order_limit() -> list[int]:
    """Per-product position limits (same as prosperity4bt LIMITS). Not enforced here — batch order
    sizes must be capped in run() so position + sum(buy qty this tick) <= limit."""
    return [POSITION_LIMIT, POSITION_LIMIT]


def _load_td(state: TradingState) -> dict[str, Any]:
    if not state.traderData:
        return {}
    try:
        return json.loads(state.traderData)
    except json.JSONDecodeError:
        return {}


def status_pepper(
    depth: OrderDepth,
    st: dict[str, Any],
) -> tuple[float, list[float], dict[str, Any]]:
    """
    Fair = mean(L2 bid, L2 ask); if L2 missing, L1 mid. If still unusable, fair is NaN (caller skips
    aggressive bid-side takes). deltas[0] = best bid change, deltas[1] = best ask change,
    deltas[2] = fair change vs previous fair.
    """
    if not depth.buy_orders or not depth.sell_orders:
        deltas = [0.0, 0.0, 0.0]
        return float("nan"), deltas, st

    fair = fair_mid(depth)
    if math.isnan(fair):
        deltas = [0.0, 0.0, 0.0]
        return float("nan"), deltas, st

    bb = float(max(depth.buy_orders.keys()))
    ba = float(min(depth.sell_orders.keys()))

    prev_bb = st.get("prev_best_bid")
    prev_ba = st.get("prev_best_ask")
    prev_fair = st.get("prev_fair")

    d0 = bb - prev_bb if prev_bb is not None else 0.0
    d1 = ba - prev_ba if prev_ba is not None else 0.0
    d2 = fair - prev_fair if prev_fair is not None else 0.0
    deltas = [d0, d1, d2]

    st["prev_best_bid"] = bb
    st["prev_best_ask"] = ba
    st["prev_fair"] = fair
    st["last_fair"] = fair
    return fair, deltas, st


def trend_assumption_update(
    st: dict[str, Any],
    fair: float,
    depth: OrderDepth,
    fair_before_tick: float | None,
) -> bool:
    """If fair dropped > FAIR_DROP_TREND_OFF vs prior tick, False until mid up-streak > MID_UP_STREAK_ON."""
    ta = bool(st.get("trend_assumption", True))
    mid = _mid_price_using_best(depth)
    prev_mid = st.get("prev_mid")

    if (
        not math.isnan(fair)
        and fair_before_tick is not None
        and fair < fair_before_tick - FAIR_DROP_TREND_OFF
    ):
        ta = False
        st["mid_up_streak"] = 0

    if not math.isnan(mid) and prev_mid is not None and not math.isnan(float(prev_mid)):
        if mid >= float(prev_mid):
            st["mid_up_streak"] = int(st.get("mid_up_streak", 0)) + 1
        else:
            st["mid_up_streak"] = 0
        if int(st.get("mid_up_streak", 0)) > MID_UP_STREAK_ON:
            ta = True

    st["prev_mid"] = mid if not math.isnan(mid) else prev_mid
    st["trend_assumption"] = ta
    return ta


def reversal_assumption_update(st: dict[str, Any], deltas: list[float]) -> list[bool]:
    """
    reversal_assumption[2]: [bid, ask]. After enough |delta_i|>=REVERSAL_LARGE_ABS samples,
    if > half didn't reverse next tick, that side becomes False.
    """
    ra = list(st.get("reversal_assumption", [True, True]))
    if len(ra) < 2:
        ra = [True, True]

    d0, d1 = deltas[0], deltas[1]

    for side, d, pend_key, tot_key, bad_key in (
        (0, d0, "pend_bid", "rev_bid_tot", "rev_bid_bad"),
        (1, d1, "pend_ask", "rev_ask_tot", "rev_ask_bad"),
    ):
        if not ra[side]:
            st[pend_key] = None
            continue
        pend = st.get(pend_key)
        if pend is not None:
            st[tot_key] = int(st.get(tot_key, 0)) + 1
            if pend * d >= 0:
                st[bad_key] = int(st.get(bad_key, 0)) + 1
            tot = int(st.get(tot_key, 0))
            bad = int(st.get(bad_key, 0))
            if tot > REVERSAL_SAMPLE_MIN and bad / tot > REVERSAL_BAD_FRAC:
                ra[side] = False
            st[pend_key] = None

        if abs(d) >= REVERSAL_LARGE_ABS:
            st[pend_key] = d

    st["reversal_assumption"] = ra
    return ra


def take_profitable(
    depth: OrderDepth,
    symbol: str,
    fair: float,
    pos: int,
    trend_ok: bool,
) -> tuple[list[Order], int]:
    """Lift asks below fair; hit bids above fair. No aggressive buys when fair is NaN.

    When trend_ok, sell-take size is capped so inventory stays >= DISABLE_PROFITABLE_SELL_TAKE.
    """
    orders: list[Order] = []
    sim = pos

    best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
    if (
        not math.isnan(fair)
        and best_ask is not None
        and best_ask < fair
        and sim < POSITION_LIMIT
    ):
        for price in sorted(depth.sell_orders.keys()):
            if price >= fair:
                break
            vol = -depth.sell_orders[price]
            if vol <= 0:
                continue
            room = POSITION_LIMIT - sim
            if room <= 0:
                break
            q = min(vol, room)
            if q <= 0:
                continue
            orders.append(Order(symbol, price, q))
            sim += q

    best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
    if best_bid is not None and best_bid > fair and sim > -POSITION_LIMIT:
        for price in sorted(depth.buy_orders.keys(), reverse=True):
            if price <= fair:
                break
            vol = depth.buy_orders[price]
            if vol <= 0:
                continue
            room = sim + POSITION_LIMIT
            if room <= 0:
                break
            q = min(vol, room)
            if trend_ok:
                max_sell_to_floor = sim - DISABLE_PROFITABLE_SELL_TAKE
                if max_sell_to_floor <= 0:
                    break
                q = min(q, max_sell_to_floor)
            if q <= 0:
                continue
            orders.append(Order(symbol, price, -q))
            sim -= q

    return orders, sim


def _alloc_by_weights(total_q: int, weights: tuple[int, ...]) -> list[int]:
    """Split ``total_q`` into integer parts proportional to ``weights`` (largest-remainder)."""
    W = sum(weights)
    n = len(weights)
    if W <= 0 or total_q <= 0:
        return [0] * n
    nums = [total_q * w for w in weights]
    qts = [nums[i] // W for i in range(n)]
    rem = total_q - sum(qts)
    order = sorted(range(n), key=lambda i: (-(nums[i] % W), i))
    for k in range(rem):
        qts[order[k]] += 1
    return qts


def initial_buy(
    depth: OrderDepth,
    symbol: str,
    deltas: list[float],
    pos0: int,
    trend_ok: bool,
    target: int,
    *,
    timestamp: int = 0,
    max_buy_qty: int | None = None,
) -> tuple[list[Order], int]:
    """Ladder at ``round(L1 mid) + {1,2,3,4}`` with weights ``INITIAL_BUY_LADDER_WEIGHTS``.

    Total qty is ``min(INITIAL_BUY_ORDER_QTY, remaining)``; allocation uses largest-remainder so
    ratios match weights at any cap. Per rung: passive clamp; dropped qty redistributed among valid
    rungs by weight; identical prices merged.
    """
    ts = f"ts={timestamp}"

    if pos0 >= target:
        logger.print(
            "initial_buy_skip",
            "reason=pos_ge_target",
            ts,
            f"pos0={pos0}",
            f"target={target}",
        )
        return [], pos0
    if not trend_ok:
        logger.print("initial_buy_skip", "reason=trend_off", ts, f"pos0={pos0}")
        return [], pos0
    d1 = deltas[1]
    if d1 >= 2:
        logger.print("initial_buy_skip", "reason=d1_ge_2", ts, f"d1={d1}", f"pos0={pos0}")
        return [], pos0
    if not depth.buy_orders or not depth.sell_orders:
        logger.print("initial_buy_skip", "reason=no_book", ts, f"pos0={pos0}")
        return [], pos0
    l1_mid = _mid_price_using_best(depth)
    if math.isnan(l1_mid):
        logger.print("initial_buy_skip", "reason=nan_l1_mid", ts, f"pos0={pos0}")
        return [], pos0
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    if max_buy_qty is not None and max_buy_qty <= 0:
        logger.print(
            "initial_buy_skip",
            "reason=no_batch_buy_headroom",
            ts,
            f"pos0={pos0}",
            f"max_buy_qty={max_buy_qty}",
        )
        return [], pos0
    remaining = min(POSITION_LIMIT - pos0, target - pos0)
    if max_buy_qty is not None:
        remaining = min(remaining, max_buy_qty)
    if remaining <= 0:
        logger.print(
            "initial_buy_skip",
            "reason=no_room",
            ts,
            f"pos0={pos0}",
            f"target={target}",
            f"remaining={remaining}",
            f"max_buy_qty={max_buy_qty}",
        )
        return [], pos0

    base = int(round(l1_mid))
    weights = INITIAL_BUY_LADDER_WEIGHTS
    offsets = INITIAL_BUY_LADDER_OFFSETS

    def passive_bid(raw: int) -> int | None:
        p = raw
        if p >= best_ask:
            p = best_ask - 1
        if p <= best_bid:
            return None
        return p

    total_q = min(INITIAL_BUY_ORDER_QTY, remaining)
    qts = _alloc_by_weights(total_q, weights)
    raw_prices = [base + off for off in offsets]
    passive_at = [passive_bid(r) for r in raw_prices]

    if all(p is None for p in passive_at):
        logger.print(
            "initial_buy_skip",
            "reason=price_not_passive",
            ts,
            f"pos0={pos0}",
            f"best_bid={best_bid}",
            f"best_ask={best_ask}",
            f"l1_mid={l1_mid}",
            f"base={base}",
        )
        return [], pos0

    merged: dict[int, int] = {}
    lost = 0
    for i, p in enumerate(passive_at):
        qi = qts[i]
        if qi <= 0:
            continue
        if p is None:
            lost += qi
        else:
            merged[p] = merged.get(p, 0) + qi

    valid_ix = [i for i in range(len(weights)) if passive_at[i] is not None]
    if lost > 0 and valid_ix:
        vw = tuple(weights[i] for i in valid_ix)
        extras = _alloc_by_weights(lost, vw)
        for j, i in enumerate(valid_ix):
            pp = passive_at[i]
            merged[pp] = merged.get(pp, 0) + extras[j]

    out = [Order(symbol, price, q) for price, q in sorted(merged.items()) if q > 0]

    logger.print(
        "initial_buy_post",
        ts,
        f"levels=mid+1,mid+2,mid+3,mid+4",
        f"weights={weights}",
        f"prices={[o.price for o in out]}",
        f"qtys={[o.quantity for o in out]}",
        f"pos0={pos0}",
        f"best_bid={best_bid}",
        f"best_ask={best_ask}",
        f"l1_mid={l1_mid}",
        f"max_buy_qty={max_buy_qty}",
        f"ib_cap={INITIAL_BUY_ORDER_QTY}",
    )
    return out, pos0


def market_make(
    depth: OrderDepth,
    symbol: str,
    pos: int,
    deltas: list[float],
    trend_ok: bool,
    *,
    max_buy_qty: int | None = None,
) -> list[Order]:
    if not depth.buy_orders or not depth.sell_orders:
        return []

    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    quote_bid = best_bid + MM_BAND_PEPPER
    quote_ask = best_ask - MM_BAND_PEPPER

    d1 = deltas[1]
    if d1 >= 2:
        sell_vol = MM_SELL_VOL_HIGH
    elif 1 < d1 < 2:
        sell_vol = MM_SELL_VOL_MID
    elif -1 <= d1 <= 1:
        sell_vol = MM_SELL_VOL_LOW
    else:
        sell_vol = MM_SELL_VOL_LOW

    sell_vol = min(sell_vol, max(0, pos + POSITION_LIMIT))
    if trend_ok and pos < DISABLE_MARKET_MAKE_SELLS:
        sell_vol = 0

    buy_sz = max(0, POSITION_LIMIT - pos)
    if max_buy_qty is not None:
        buy_sz = min(buy_sz, max(0, max_buy_qty))

    out: list[Order] = []
    if buy_sz > 0:
        out.append(Order(symbol, int(quote_bid), buy_sz))
    if sell_vol > 0:
        out.append(Order(symbol, int(quote_ask), -sell_vol))
    return out


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
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                buyer = trade.buyer if trade.buyer is not None else ""
                seller = trade.seller if trade.seller is not None else ""
                compressed.append(
                    [trade.symbol, int(trade.price), int(trade.quantity), buyer, seller, int(trade.timestamp)]
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
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

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

        td = _load_td(state)
        pepper_st: dict[str, Any] = dict(td.get("pepper", {}))

        ash_fair, mids = _ash_fair_and_mids(state, td)
        if DAVIS_FAIR_MA_WINDOW > 0:
            td["mids"] = mids

        depth_p = state.order_depths.get(INTARIAN_PEPPER_ROOT)
        if depth_p is not None:
            fair_before = pepper_st.get("prev_fair")
            if isinstance(fair_before, str):
                try:
                    fair_before = float(fair_before)
                except ValueError:
                    fair_before = None
            fair, deltas, pepper_st = status_pepper(depth_p, pepper_st)
            trend_ok = trend_assumption_update(pepper_st, fair, depth_p, fair_before)
            reversal_assumption_update(pepper_st, deltas)

            pos0 = state.position.get(INTARIAN_PEPPER_ROOT, 0)
            orders: list[Order] = []
            takes, sim = take_profitable(
                depth_p, INTARIAN_PEPPER_ROOT, fair, pos0, trend_ok
            )
            orders.extend(takes)
            buy_vol_takes = sum(o.quantity for o in takes if o.quantity > 0)
            rem_buy = max(0, POSITION_LIMIT - pos0 - buy_vol_takes)
            ib, _ = initial_buy(
                depth_p,
                INTARIAN_PEPPER_ROOT,
                deltas,
                pos0,
                trend_ok,
                INITIAL_BUY_INVENTORY_TARGET,
                timestamp=state.timestamp,
                max_buy_qty=rem_buy,
            )
            orders.extend(ib)
            ib_buy = sum(o.quantity for o in ib if o.quantity > 0)
            rem_mm_buy = max(0, rem_buy - ib_buy)
            if ib_buy > 0:
                rem_mm_buy = 0
            mm = market_make(
                depth_p,
                INTARIAN_PEPPER_ROOT,
                pos0,
                deltas,
                trend_ok,
                max_buy_qty=rem_mm_buy,
            )
            for o in mm:
                if o.quantity > 0:
                    logger.print(
                        "pepper_mm_buy",
                        f"ts={state.timestamp}",
                        f"price={o.price}",
                        f"qty={o.quantity}",
                        f"pos0={pos0}",
                        f"trend_ok={trend_ok}",
                    )
            orders.extend(mm)
            result[INTARIAN_PEPPER_ROOT] = orders
        else:
            result[INTARIAN_PEPPER_ROOT] = []

        td["pepper"] = pepper_st
        trader_data = json.dumps(td, separators=(",", ":"))

        for product in state.order_depths:
            if product == ASH_COATED_OSMIUM:
                result[product] = self._osmium_orders(state, ash_fair)
            elif product != INTARIAN_PEPPER_ROOT:
                result[product] = []

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _osmium_orders(self, state: TradingState, fair_value: int) -> list[Order]:
        depth = state.order_depths[ASH_COATED_OSMIUM]
        pos = state.position.get(ASH_COATED_OSMIUM, 0)
        orders: list[Order] = []
        if fair_value == 0:
            return orders

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
        buy_vol_takes = sum(o.quantity for o in take_orders if o.quantity > 0)
        sell_vol_takes = sum(-o.quantity for o in take_orders if o.quantity < 0)
        # Batch limit: pos + sum(buy qty) <= 80 and pos - sum(sell qty) >= -80 (prosperity4bt enforce_limits).
        rem_buy_batch = max(0, POSITION_LIMIT - pos - buy_vol_takes)
        rem_sell_batch = max(0, pos + POSITION_LIMIT - sell_vol_takes)

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

        mm_buy_sz = min(max(0, POSITION_LIMIT - pos_after_takes), rem_buy_batch)
        mm_sell_sz = min(max(0, pos_after_takes + POSITION_LIMIT), rem_sell_batch)

        mm_orders: list[Order] = []
        if mm_buy_sz > 0:
            mm_orders.append(Order(ASH_COATED_OSMIUM, mm_bid, mm_buy_sz))
        if mm_sell_sz > 0:
            mm_orders.append(Order(ASH_COATED_OSMIUM, mm_ask, -mm_sell_sz))

        orders.extend(take_orders)
        orders.extend(mm_orders)
        return orders
