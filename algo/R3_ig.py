from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
from collections import deque
import string, json, math, statistics
import numpy as np

# MATH FOR BLACK SCHOLES
from math import floor, ceil, log, sqrt, exp
from statistics import NormalDist

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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

'''
DEFAULT CLASSES
'''

class Product():
    '''
    SECTION 1 - Gather Information
    '''
    def __init__(self, product: str, limit: int, state: TradingState):
        # information in the trading state
        self.state = state
        self.traderData = state.traderData
        self.timestamp = state.timestamp
        self.listings = state.listings
        self.order_depth = state.order_depths.get(product, None)
        self.own_trades = state.own_trades
        self.market_trades = state.market_trades
        self.position = state.position
        self.observations = state.observations

        # information about the current product
        self.product = product
        self.limit = limit
        self.position = state.position[product] if product in state.position else 0
        self.nbuy = 0
        self.nsell = 0
        self.orders: List[Order] = [] # how to submit orders!

    # replace trading state; reset other needed information
    def reset_state(self, new_state: TradingState):
        # information in the trading state
        self.state = new_state
        self.traderData = new_state.traderData
        self.timestamp = new_state.timestamp
        self.listings = new_state.listings
        self.order_depth = new_state.order_depths.get(self.product, None)
        self.own_trades = new_state.own_trades
        self.market_trades = new_state.market_trades
        self.position = new_state.position
        self.observations = new_state.observations

        # information about the current product
        self.position = new_state.position[self.product] if self.product in new_state.position else 0
        self.nbuy = 0
        self.nsell = 0
        self.orders: List[Order] = []

    # our current position    
    def active_position(self):
        return self.position + self.nbuy - self.nsell  

    # total volume on the buy / sell side
    def orderbook_buy_size(self):
        return sum(self.order_depth.buy_orders.values())
    def orderbook_sell_size(self):
        return -sum(self.order_depth.sell_orders.values())
    
    # maximum amt to buy / sell, given our positions
    def limit_buy_orders(self):
        return self.limit - self.position - self.nbuy 
    def limit_sell_orders(self):
        return self.limit + self.position - self.nsell
    
    # maximum amt to buy / sell given info about ourself + the market
    def max_buy_orders(self):
        return min(self.limit_buy_orders(), self.orderbook_sell_size())
    def max_sell_orders(self):
        return min(self.limit_sell_orders(), self.orderbook_buy_size())

    # best / worst for both bid / ask
    def best_bid(self):
        if len(self.order_depth.buy_orders) > 0:
            return max(self.order_depth.buy_orders.keys())
        else:
            return math.nan
    def best_ask(self):
        if len(self.order_depth.sell_orders) > 0:
            return min(self.order_depth.sell_orders.keys())
        else:
            return math.nan
    def worst_bid(self):
        if len(self.order_depth.buy_orders) > 0:
            return min(self.order_depth.buy_orders.keys())
        else:
            return math.nan
    def worst_ask(self):
        if len(self.order_depth.sell_orders) > 0:
            return max(self.order_depth.sell_orders.keys())
        else:
            return math.nan

    # calculate mid prices using best / worst (there are reasons to use one over the other!)
    def mid_price_using_best(self):
        return (self.best_bid() + self.best_ask()) / 2
    def mid_price_using_worst(self):
        return (self.worst_bid() + self.worst_ask()) / 2
    
    '''
    SECTION 2 - Interact with the Market
    '''
    # default buy / sell methods
    def buy(self, price: int, quantity: int, print: bool=False):
        if print:
            logger.print("Buy Order: ", price, quantity)     
        if quantity > self.limit_buy_orders():
            logger.print("Buy Order: ", price, quantity, " exceeds max buy orders")
        elif quantity > 0 and quantity <= self.limit_buy_orders():
            self.orders.append(Order(self.product, int(price), quantity))
            self.nbuy += quantity
    def sell(self, price: int, quantity: int, print: bool=False):
        if print:
            logger.print("Sell Order: ", price, quantity)
        if quantity > self.limit_sell_orders():
            logger.print("Sell Order: ", price, quantity, " exceeds max sell orders")
        elif quantity > 0 and quantity <= self.limit_sell_orders():
            self.orders.append(Order(self.product, int(price), -quantity))
            self.nsell += quantity

    # methods to buy / sell everything
    def full_buy(self, quantity: int):
        q = quantity
        for price in sorted(self.order_depth.sell_orders.keys()):
            if q == 0:
                break
            available = -self.order_depth.sell_orders[price]
            buy_quantity = min(self.max_buy_orders(), available)
            if buy_quantity > 0:
                self.buy(price, buy_quantity)
                q -= buy_quantity
    def full_sell(self, quantity: int):
        q = quantity
        for price in sorted(self.order_depth.buy_orders.keys(), reverse = True):
            if q == 0:
                break
            available = self.order_depth.buy_orders[price]
            sell_quantity = min(q, available)
            if sell_quantity > 0:
                self.sell(price, sell_quantity)
                q -= sell_quantity
    
    # cancel all orders, all / buy / sell
    def cancel_orders(self):
        self.orders = []
        self.nsell = 0
        self.nbuy = 0   
    def cancel_buy_orders(self):
        self.orders = [order for order in self.orders if order.quantity < 0]
        self.nsell = 0
        self.nbuy = 0
    def cancel_sell_orders(self):
        self.orders = [order for order in self.orders if order.quantity > 0]
        self.nsell = 0
        self.nbuy = 0

    '''
    SECTION 3 - Strategies
    '''
    
    def take(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        for price in sorted(self.order_depth.sell_orders.keys()):
            if price >= fair_val - edge:
                break
            available = -self.order_depth.sell_orders[price]
            quantity = min(self.max_buy_orders(), available)
            if quantity > 0:
                self.buy(price, quantity)
        
        for price in sorted(self.order_depth.buy_orders.keys(), reverse = True):
            if price <= fair_val + edge:
                break
            available = self.order_depth.buy_orders[price]
            quantity = min(self.max_sell_orders(), available)
            if quantity > 0:
                self.sell(price, quantity)

    def clear(
        self,
        fair_val: int,
    ):
        if self.active_position() > 0:
            self.sell(fair_val, self.active_position())
        elif self.active_position() < 0:
            self.buy(fair_val, -self.active_position())

    def make(
        self,
        fair_val: int,
        edge: float,
    ):
        self.buy(fair_val - edge, self.max_buy_orders())
        self.sell(fair_val + edge, self.max_sell_orders())

    def make_balanced(
        self,
        fair_val: int,
        edge: float,
    ):
        if not (self.active_position() > 0 and fair_val - edge >= self.best_bid()):
            self.buy(fair_val - edge, self.max_buy_orders())
        if not (self.active_position() < 0 and fair_val + edge <= self.best_ask()):
            self.sell(fair_val + edge, self.max_sell_orders())

    def take_clear_make(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        self.take(fair_val, edge)
        self.clear(fair_val)
        self.make(fair_val, edge)
    
    '''
    SECTION 4 - Non-Implemented Strategies (runtime polymorphism)
    '''
    def fair_val(self):   
        raise NotImplementedError()
        ...

    def strategy(self):
        raise NotImplementedError()
        ...

class RollingZ():
    def __init__(self, z_th: float, window: int, fixed_mean: float = math.nan, fixed_std: float = math.nan):
        self.premiums = np.array([], dtype = float)
        self.z_th = z_th
        self.window = window
        self.fixed_mean = fixed_mean
        self.fixed_std = fixed_std

    def add(self, new_val: float):
        self.premiums = np.append(self.premiums, new_val)
        if len(self.premiums) > self.window:
            self.premiums = np.delete(self.premiums, 0)
            assert(len(self.premiums) == self.window)
        assert(len(self.premiums) != 0)

    def mean(self):
        return self.fixed_mean if not math.isnan(self.fixed_mean) else np.mean(self.premiums)

    def std(self):
        return self.fixed_std if not math.isnan(self.fixed_std) else np.std(self.premiums)
    
    def most_recent(self):
        return self.premiums[-1]

    # 1 means buy, 0 means do nothing, -1 means sell (sign indicates position direction)
    def signal(self):
        if len(self.premiums) != self.window or np.std(self.premiums) == 0:
            return 0
        
        z_score = (self.most_recent() - self.mean()) / self.std()
        logger.print(f"z_score = {z_score}")
        if z_score < -self.z_th:
            return 1
        elif z_score > self.z_th:
            return -1
        else:
            return 0

class Arbitrage(Product):
    def __init__(self, 
                 product: str, limit: int, state: TradingState, 
                 lhs: List[Product], lhs_scalars: List[int], 
                 rhs: List[Product], rhs_scalars: List[int], 
                 z_th: float, window: int, 
                 fixed_mean: float = math.nan, fixed_std: float = math.nan):
        super().__init__(product, limit, state)
        self.rolling_z = RollingZ(z_th = z_th, window = window, fixed_mean = fixed_mean, fixed_std = fixed_std)
        self.lhs = lhs
        self.lhs_scalars = lhs_scalars
        self.rhs = rhs
        self.rhs_scalars = rhs_scalars
        self.executed = False

    def fair_val(self):
        return self.rolling_z.mean()

    def strategy(self):
        # this represents price of (LHS - RHS)
        difference = 0

        # LToR means buy LHS sell RHS
        maxLToR = 1000000
        maxRToL = 1000000
        for product, amt in zip(self.lhs, self.lhs_scalars):
            maxLToR = min(maxLToR, product.max_buy_orders() // amt)
            maxRToL = min(maxRToL, product.max_sell_orders() // amt)
            difference += product.mid_price_using_best() * amt

        for product, amt in zip(self.rhs, self.rhs_scalars):
            maxLToR = min(maxLToR, product.max_sell_orders() // amt)
            maxRToL = min(maxRToL, product.max_buy_orders() // amt)
            difference -= product.mid_price_using_best() * amt

        self.rolling_z.add(difference)
        signal = self.rolling_z.signal()
        logger.print(f"signal = {signal}")

        if signal == 1:
            self.executed = True
            # RHS is more expensive than LHS -> buy LHS, sell RHS
            if maxLToR > 0:
                logger.print(f"maxLToR = {maxLToR}")
                for product, amt in zip(self.lhs, self.lhs_scalars):
                    product.buy(product.worst_ask(), amt * maxLToR)
                for product, amt in zip(self.rhs, self.rhs_scalars):
                    product.sell(product.worst_bid(), amt * maxLToR)
        elif signal == -1:
            self.executed = True
            # LHS is more expensive than RHS -> buy RHS, sell LHS
            if maxRToL > 0:
                logger.print(f"maxRToL = {maxRToL}")
                for product, amt in zip(self.rhs, self.rhs_scalars):
                    product.buy(product.worst_ask(), amt * maxRToL)
                for product, amt in zip(self.lhs, self.lhs_scalars):
                    product.sell(product.worst_bid(), amt * maxRToL)
        else:
            # rebalance position
            for product, amt in zip(self.lhs, self.lhs_scalars):
                product.mm_undercut_balanced(product.fair_val(), 1)
            for product, amt in zip(self.rhs, self.rhs_scalars):
                product.mm_undercut_balanced(product.fair_val(), 1)

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)
    
    @staticmethod
    def delta_put(spot, strike, time_to_expiry, volatility):
        return BlackScholes.delta_call(spot, strike, time_to_expiry, volatility) - 1

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility_call(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.0001
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility
    
    @staticmethod
    def implied_volatility_put(
        put_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.0001
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_put(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - put_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    @staticmethod
    def moneyness(spot, strike, time_to_expiry):
        return log(spot / strike) / sqrt(time_to_expiry)

class MeanRevOption(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState, 
                 is_call: bool, strike: int, tte: float, underlying: Product,
                 underlying_z_th: float, underlying_window: int, 
                 iv_z_th: float, iv_window: int, 
                 delta_z_th: float, delta_window: int):
        super().__init__(symbol, limit, state)
        self.is_call = is_call
        self.strike = strike
        self.tte = tte # time to expiry in years
        self.underlying = underlying
        self.underlying_prices = RollingZ(underlying_z_th, underlying_window)
        self.underlying_bought = 0
        self.underlying_sold = 0
        self.ivs = RollingZ(iv_z_th, iv_window)
        self.deltas = RollingZ(delta_z_th, delta_window)
    
    def update_tte(self, new_tte: float):
        self.tte = new_tte

    def fair_val(self):
        if len(self.ivs.premiums) != self.ivs.window:
            # when we can't trade, we are NOT bingqilin
            logger.print("NOT bingqilin")
            return
        underlying_mid = self.underlying.fair_val()
        avg_vol = self.ivs.mean()
        std_vol = self.ivs.std()
        if self.is_call:
            cur_prices = [round(BlackScholes.black_scholes_call(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        else:
            cur_prices = [round(BlackScholes.black_scholes_put(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        return cur_prices[1]

    def strategy(self):
        # add information to the RollingZ objects
        underlying_mid = self.underlying.fair_val()
        cur_mid = self.mid_price_using_best()
        if self.is_call:
            option_iv = BlackScholes.implied_volatility_call(cur_mid, underlying_mid, self.strike, self.tte)
            option_delta = BlackScholes.delta_call(underlying_mid, self.strike, self.tte, option_iv)
        else:
            option_iv = BlackScholes.implied_volatility_put(cur_mid, underlying_mid, self.strike, self.tte)
            option_delta = BlackScholes.delta_put(underlying_mid, self.strike, self.tte, option_iv)
        
        # cap ivs getting added (there's something causing it to spike like CRAZY)
        if not (0.01 <= option_iv <= 0.99):
            if len(self.ivs.premiums) != self.ivs.window:
                return
            option_iv = max(self.ivs.mean() - 2 * self.ivs.std(), 
                            min(option_iv, 
                                self.ivs.mean() + 2 * self.ivs.std(),
                            )
                        )
        
        self.underlying_prices.add(underlying_mid)
        self.ivs.add(round(option_iv, 6))
        self.deltas.add(round(option_delta, 6))
        if len(self.ivs.premiums) != self.ivs.window:
            # when we can't trade, we are NOT bingqilin
            logger.print("NOT bingqilin")
            return

        # make prices; round to 6 places for readability
        avg_vol = self.ivs.mean()
        std_vol = self.ivs.std()
        if self.is_call:
            cur_prices = [round(BlackScholes.black_scholes_call(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        else:
            cur_prices = [round(BlackScholes.black_scholes_put(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        
        # leave if the market is not volatile enough
        if cur_prices[2] - cur_prices[0] < 1.0:
            return
        
        # we market making ts
        fair_value = cur_prices[1]
        if math.isnan(fair_value):
            return
        bid = int(math.floor(fair_value - 0.01))
        ask = int(math.ceil(fair_value + 0.01))
        old_position = self.active_position()

        # DEBUG
        # logger.print("cur_prices:", cur_prices)
        # logger.print("cur_mid:", -1 if math.isnan(self.mid_price_using_best()) else self.mid_price_using_best())
        # if not math.isnan(self.mid_price_using_best()) and abs(self.mid_price_using_best() - fair_value) > 5:
        #     logger.print("THE BAD IS HERE")

        # phase 1 - take
        self.take(fair_value)
        
        # phase 2 - clear
        bid_size = self.limit_buy_orders()
        ask_size = self.limit_sell_orders()
        if bid == ask:
            if bid_size > ask_size:
                self.buy(bid, bid_size)
            elif ask_size > bid_size:
                self.sell(ask, ask_size)
        else:
            self.buy(bid, bid_size)
            self.sell(ask, ask_size)

        # phase 3 (make) is skipped b/c it does nothing...

        # delta hedge
        position_change = self.active_position() - old_position
        avg_delta = self.deltas.mean()
        if self.is_call:
            if position_change > 0:
                self.full_sell(int(ceil(position_change * avg_delta)))
            elif position_change < 0:
                self.full_buy(int(floor(-position_change * avg_delta)))
        else:
            if position_change > 0:
                self.full_buy(int(ceil(position_change * -avg_delta)))
            elif position_change < 0:
                self.full_sell(int(floor(-position_change * -avg_delta)))

class IVScalperOption(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState, 
                 is_call: bool, strike: int, tte: float, underlying: Product):
        super().__init__(symbol, limit, state)
        self.is_call = is_call
        self.strike = strike
        self.tte = tte # time to expiry in years
        self.underlying = underlying
        self.theos = RollingZ(1, 1000)

        # with filtering
        self.a = 0.12828680415426924
        self.b = 0.006339066632356327
        self.c = 0.2723988924431878

        # without filtering
        # self.a = 0.06422522746659134
        # self.b = -0.10814309837562446
        # self.c = 0.2363099052068178
    
    def update_tte(self, new_tte: float):
        self.tte = new_tte

    def implied_volatility(self):
        moneyness = BlackScholes.moneyness(self.underlying.mid_price_using_best(), self.strike, self.tte)
        return self.a * moneyness ** 2 + self.b * moneyness + self.c

    def fair_val(self):
        return BlackScholes.black_scholes_call(self.underlying.mid_price_using_best(), self.strike, self.tte, self.implied_volatility())
    
    def strategy(self):
        fv = self.fair_val()
        self.theos.add(fv)
        signal = self.theos.signal()
        if signal == -1:
            logger.print(f"product = {self.product}")
            logger.print(f"self.theos.mean() = {self.theos.mean()}, self.theos.most_recent() = {self.theos.most_recent()}")
            self.sell(int(ceil(fv + 0.1)), self.max_sell_orders())
        elif signal == 1:
            logger.print(f"product = {self.product}")
            logger.print(f"self.theos.mean() = {self.theos.mean()}, self.theos.most_recent() = {self.theos.most_recent()}")
            self.buy(int(floor(fv - 0.1)), self.max_buy_orders())
        

'''
PRODUCT CLASSES
'''

class VelvetFruitExtract(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass # self.take_clear_make(self.fair_val(), 3)

class VEV(IVScalperOption):
    def __init__(self, symbol: str, limit: int, state: TradingState, 
                 is_call: bool, strike: int, tte: float, underlying: Product):
        super().__init__(symbol, limit, state, is_call, strike, tte, underlying)
    
    def fair_val(self):
        return super().fair_val()
    
    def strategy(self):
        super().strategy()

class Hydrogel(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        fair_val = self.mid_price_using_best()
        making_th = 8
        if (not math.isnan(self.best_bid())) and (not math.isnan(self.best_ask())):
            bid = min(fair_val - making_th, self.best_bid() + 1)
            ask = max(fair_val + making_th, self.best_ask() - 1)
            self.make_balanced((bid + ask) / 2, (ask - bid) / 2)

'''
TRADING EXECUTION
'''

product_instances = []

class Trader:
    # indicates if product_instances has been filled
    turned_on = False

    # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
    def run(self, state: TradingState):
        result = {}
        traderData = ""
        cur_day = 3
        cur_tte = ((8 - cur_day) / 365) - (state.timestamp / 365e6) # SET THIS!

        if not Trader.turned_on:
            # initiate the products / arbitrages
            product_instances.append(VelvetFruitExtract("VELVETFRUIT_EXTRACT", 200, state))
            for st in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
                product_instances.append(VEV(
                    "VEV_" + str(st), 300, state, True,
                    st, cur_tte, product_instances[0]
                ))
            # product_instances.append(Hydrogel("HYDROGEL_PACK", 200, state))

            # turn on the trading unit; the products have been populated!
            Trader.turned_on = True
        else:
            # reset ALL the states (can help if multiple product states are entangled)
            for instance in product_instances:
                instance.reset_state(state)

        # update tte for options
        for instance in product_instances:
            if isinstance(instance, MeanRevOption) or isinstance(instance, IVScalperOption):
                instance.update_tte(cur_tte)

        # after ALL instantiating or resetting is done, then execute strategies
        for instance in product_instances:
            if isinstance(instance, Product):
                instance.strategy()
                result[instance.product] = instance.orders
                logger.print("Orders for ", instance.product, ": ", instance.orders)

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData