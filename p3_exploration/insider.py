from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
from collections import deque
import string, json, math, statistics
import numpy as np

# MATH FOR BLACK SCHOLES
from math import floor, ceil, log, sqrt, exp
from statistics import NormalDist

'''
NOTE: This file will contain all products except Macarons from Prosperity 3. 
'''

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 10000 # was 3750

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
        for price, volume in self.order_depth.sell_orders.items():
            if volume < 0:
                buy_vol = min(q, min(self.limit_buy_orders(), -volume))
                self.buy(price, buy_vol)
                q -= buy_vol
                if q <= 0 or self.limit_buy_orders() <= 0:
                    break
    def full_sell(self, quantity: int):
        q = quantity
        for price, volume in self.order_depth.buy_orders.items():
            if volume > 0:
                sell_vol = min(q, min(self.limit_sell_orders(), volume))
                self.sell(price, sell_vol)
                q -= sell_vol
                if q <= 0 or self.limit_sell_orders() <= 0:
                    break
    
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
    # balance position back to zero using market participants' prices
    def market_take(self, fair_val: float, edge: float = 0):
        bid_val = fair_val - edge
        ask_val = fair_val + edge

        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > ask_val or (bid_price == ask_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == ask_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    self.sell(bid_price, sell_vol)

        for ask_price, ask_vol in self.order_depth.sell_orders.items():
            ask_vol *= -1
            if ask_price < bid_val or (ask_price == bid_val and self.active_position() < 0):
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if ask_price == bid_val:
                    buy_vol = min(buy_vol, -self.active_position())
                if buy_vol > 0:
                    self.buy(ask_price, buy_vol)

    # places a bid / ask like normal
    def market_make(
        self,
        buy_price: int,
        sell_price: int
    ):
        self.buy(buy_price, self.max_buy_orders())
        self.sell(sell_price, self.max_sell_orders())
    
    # market-making under the condition of starting at [fv - edge, fv + edge] and expanding outwards
    def mm_undercut(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1
        self.market_make(mm_buy, mm_sell)

    # same as mm_undercut but also try to move position towards zero if position != 0
    def mm_undercut_balanced(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default = fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default = fair_val + edge + 1) - 1
        if not (self.active_position() > 0 and mm_buy >= self.best_bid()):
            self.buy(mm_buy, self.max_buy_orders())
        if not (self.active_position() < 0 and mm_sell <= self.best_ask()):
            self.sell(mm_sell, self.max_sell_orders())
    
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
        if np.std(self.premiums) == 0 or len(self.premiums) != self.window:
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

class Option(Product):
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
        self.underlying_bought = 0
        self.underlying_sold = 0

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
            option_iv = max(self.ivs.mean() - 2 * self.ivs.std(), min(option_iv, self.ivs.mean() + 2 * self.ivs.std()))
        
        self.underlying_prices.add(underlying_mid)
        self.ivs.add(round(option_iv, 6))
        self.deltas.add(round(option_delta, 6))
        if len(self.ivs.premiums) != self.ivs.window:
            # when we can't trade, we are NOT bingqilin
            logger.print("NOT bingqilin")
            return

        # DEBUG - check for bad IVs
        # # Define reasonable IV bounds
        # MIN_IV = 0.0       # IV can't be negative
        # MAX_IV = 0.5       # Adjust depending on market
        # # Check the array
        # bad_indices = np.where((self.ivs.premiums < MIN_IV) | (self.ivs.premiums > MAX_IV))[0]
        # # Assert that there are no bad values
        # assert len(bad_indices) == 0, f"Found {len(bad_indices)} invalid IVs at indices {bad_indices}, values: {self.ivs.premiums[bad_indices]}"

        # make prices; round to 6 places for readability
        avg_vol = self.ivs.mean()
        std_vol = self.ivs.std()
        if self.is_call:
            cur_prices = [round(BlackScholes.black_scholes_call(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        else:
            cur_prices = [round(BlackScholes.black_scholes_put(underlying_mid, self.strike, self.tte, avg_vol + i * std_vol), 6) for i in range(-1, 2)]
        
        # market make on it
        fair_value = cur_prices[1]
        bid = int(math.floor(fair_value + 0.01))
        ask = int(math.ceil(fair_value - 0.01))

        # DEBUG - check for valid iv values
        # logger.print("iv:", round(option_iv, 6))
        # logger.print("avg_iv:", round(avg_vol, 6))
        # logger.print("fair_value:", round(fair_value, 6))
        # logger.print("market:", bid, "@", ask)
        # if not math.isnan(self.best_ask()) and self.best_ask() < bid:
        #     logger.print("we're buying")
        # if not math.isnan(self.best_bid()) and ask < self.best_bid():
        #     logger.print("we're selling")

        # if not volatile enough, just leave the market you CLOWN
        if (cur_prices[2] - cur_prices[0]) < 1.0:
            return
        
        # keep track of information
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if math.isnan(best_bid) or math.isnan(best_ask):
            logger.print("missing best bid/ask")
            return
        bid_size = self.limit_buy_orders()
        ask_size = self.limit_sell_orders()
        max_spread = max(1, math.floor(fair_value * 0.03))
        old_position = self.active_position()

        # check if we are crossing markets with best_ask
        for market_ask, market_amount in self.order_depth.sell_orders.items():
            if bid > market_ask:
                # eat their market then take it over
                eat_order_size = abs(min(bid_size, abs(market_amount)))
                self.buy(market_ask, eat_order_size)
                    
                # place bid above best bid
                if best_bid is not None:
                    bid = best_bid + 1
                else:
                    bid = int(math.floor(fair_value - max_spread))

                # place ask at maximum dist from fair value
                ask = int(math.ceil(max(fair_value + max_spread, ask)))
                                
        # check if we are crossing with best_bid
        for market_bid, market_amount in self.order_depth.buy_orders.items():
            if ask < market_bid:
                # eat their market then take it over
                eat_order_size = abs(min(ask_size, abs(market_amount)))
                self.sell(market_bid, eat_order_size)
                
                # place ask below best ask
                if best_ask is not None: 
                    ask = best_ask - 1
                else:
                    ask = int(math.ceil(ask + max_spread))
                
                # place bid at maximum dist from fair value
                bid = int(math.floor(max(fair_value - max_spread, bid)))
        
        # rebalance position
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

        # delta hedge
        position_change = self.active_position() - old_position
        avg_delta = self.deltas.mean()
        if self.is_call:
            if position_change > 0:
                self.underlying.sell(self.underlying.mid_price_using_best(), int(ceil(position_change * avg_delta)))
            elif position_change < 0:
                self.underlying.buy(self.underlying.mid_price_using_best(), int(floor(-position_change * avg_delta)))
        else:
            if position_change > 0:
                self.underlying.buy(self.underlying.mid_price_using_best(), int(ceil(position_change * -avg_delta)))
            elif position_change < 0:
                self.underlying.sell(self.underlying.mid_price_using_best(), int(floor(-position_change * -avg_delta)))

'''
PRODUCT CLASSES
'''

class Resin(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        self.market_take(self.fair_val())
        self.mm_undercut(self.fair_val(), 8)

class Kelp(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        self.market_take(self.fair_val())
        self.mm_undercut(self.fair_val(), 2)

class SquidInk(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
        self.recent_prices = RollingZ(20, 10)
    
    def fair_val(self):
        return self.mid_price_using_best()

    def strategy(self):
        # add data
        self.recent_prices.add(self.fair_val())

        # detect Olivia's signal
        market_trades = self.state.market_trades.get(self.product, [])
        market_trades_olivia = [i for i in market_trades if (i.buyer == "Olivia" or i.seller == "Olivia")]
        if len(market_trades_olivia) > 0 and abs(market_trades_olivia[-1].timestamp - self.timestamp) <= 100:
            # DEBUG
            cur_trade_olivia = market_trades_olivia[-1]
            logger.print("found")
            logger.print("olivia price:", cur_trade_olivia.price)
            logger.print("olivia's side:", ("buy" if cur_trade_olivia.buyer == "Olivia" else "sell"))
            
            # actual logic - time to go full monke
            # (analysis reveals that these are 21 ~ 28; in general it's rare to be above 40)
            if cur_trade_olivia.buyer == "Olivia":
                self.full_buy(self.orderbook_sell_size())
                logger.print("sell orderbook size =", self.orderbook_sell_size())
            else:
                self.full_sell(self.orderbook_buy_size())
                logger.print("buy orderbook size =", self.orderbook_buy_size())
            return

        # yolo. (but only on huge price change)
        if len(self.recent_prices.premiums) >= 2 and self.recent_prices.std() > 20:
            if self.fair_val() > self.recent_prices.premiums[-2] + 10:
                self.full_sell(self.orderbook_buy_size())
            elif self.fair_val() < self.recent_prices.premiums[-2] - 10:
                self.full_buy(self.orderbook_sell_size())

class Croissant(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass # self.mm_undercut_balanced(self.fair_val(), 1)

class Jam(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass # self.mm_undercut_balanced(self.fair_val(), 1)

class Djembe(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass # self.mm_undercut_balanced(self.fair_val(), 1)

class Basket1(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass

class Basket2(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        pass

class Rock(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        self.market_take(self.fair_val())
        self.mm_undercut_balanced(self.fair_val(), 5)

class RockVoucher(Option):
    def __init__(self, symbol, limit, state, is_call, strike, tte, underlying, underlying_z_th, underlying_window, iv_z_th, iv_window, delta_z_th, delta_window):
        super().__init__(symbol, limit, state, is_call, strike, tte, underlying, underlying_z_th, underlying_window, iv_z_th, iv_window, delta_z_th, delta_window)
    
    def fair_val(self):
        return super().fair_val()
    
    def strategy(self):
        super().strategy()

class Macaron(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
        self.conversions = 0
    
    def fair_val(self):
        pass
    
    def strategy(self):
        # reset conversions amount
        self.conversions = 0

        # convert anything by buying local
        if self.active_position() < 0:
            self.conversions = min(-self.active_position(), 10) # 10 at a time

        # get observation info
        obs = self.state.observations.conversionObservations.get("MAGNIFICENT_MACARONS", None)
        if obs is None:
            return
        island_ask = obs.askPrice
        transport_fee = obs.transportFees
        import_tariff = obs.importTariff

        # get local trading price and arbitrage
        ask_val = math.ceil(island_ask + transport_fee + import_tariff)
        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > ask_val or (bid_price == ask_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == ask_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    self.sell(bid_price, sell_vol)

'''
TRADING EXECUTION
'''

product_instances = {}

class Trader:
    # indicates if product_instances has been filled
    turned_on = False

    # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
    def run(self, state: TradingState):
        result = {}
        traderData = ""
        # SET THIS CORRECTLY
        cur_day = 3
        cur_tte = ((8 - cur_day) / 365) - (state.timestamp / 365e6)

        if not Trader.turned_on:
            # initiate the products / arbitrages
            # NOTE: exclude Macarons, as there is no backtesting for conversions

            # DAY 1
            # product_instances["RAINFOREST_RESIN"] = Resin("RAINFOREST_RESIN", 50, state)
            # product_instances["KELP"] = Kelp("KELP", 50, state)
            # product_instances["SQUID_INK"] = SquidInk("SQUID_INK", 50, state)

            # DAY 2
            # product_instances["CROISSANTS"] = Croissant("CROISSANTS", 250, state)
            # product_instances["JAMS"] = Jam("JAMS", 350, state)
            # product_instances["DJEMBES"] = Djembe("DJEMBES", 60, state)
            # product_instances["PICNIC_BASKET1"] = Basket1("PICNIC_BASKET1", 60, state)
            # product_instances["PICNIC_BASKET2"] = Basket2("PICNIC_BASKET2", 100, state)
            
            # product_instances["PB1_WITH_PB2"] = Arbitrage(
            #     "PB1_WITH_PB2", 1000000000, state,
            #     [product_instances["PICNIC_BASKET1"]],
            #     [1],
            #     [product_instances["PICNIC_BASKET2"], 
            #      product_instances["CROISSANTS"], 
            #      product_instances["JAMS"], 
            #      product_instances["DJEMBES"]],
            #     [1, 2, 1, 1],
            #     3.6, 100
            # )

            # OLD CODE
            # product_instances["ARBITRAGE_1"] = Arbitrage(
            #     "ARBITRAGE_1", 1000000000, state,
            #     [product_instances["PICNIC_BASKET1"]], 
            #     [1],
            #     [product_instances["CROISSANTS"], product_instances["JAMS"], product_instances["DJEMBES"]], 
            #     [6, 3, 1],
            #     1, 30, 48, 85
            # )
            # product_instances["ARBITRAGE_2"] = Arbitrage(
            #     "ARBITRAGE_2", 1000000000, state,
            #     [product_instances["PICNIC_BASKET2"]], 
            #     [1],
            #     [product_instances["CROISSANTS"], product_instances["JAMS"]],
            #     [4, 2],
            #     1.5, 30, 30, 60
            # )

            # DAY 3
            product_instances["VOLCANIC_ROCK"] = Rock("VOLCANIC_ROCK", 400, state)
            product_instances["VOLCANIC_ROCK_VOUCHER_9500"]  = RockVoucher("VOLCANIC_ROCK_VOUCHER_9500",  200, state, True, 9500,  cur_tte, product_instances["VOLCANIC_ROCK"], 20, 1000, 20, 20, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_9750"]  = RockVoucher("VOLCANIC_ROCK_VOUCHER_9750",  200, state, True, 9750,  cur_tte, product_instances["VOLCANIC_ROCK"], 20, 1000, 20, 20, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10000"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10000", 200, state, True, 10000, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 1000, 20, 20, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10250"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10250", 200, state, True, 10250, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 1000, 20, 20, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10500"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10500", 200, state, True, 10500, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 1000, 20, 20, 20, 10)

            # turn on the trading unit; the products have been populated!
            Trader.turned_on = True
        else:
            # reset ALL the states (can help if multiple product states are entangled)
            for product, instance in product_instances.items():
                instance.reset_state(state)

        # update tte for options
        for product, instance in product_instances.items():
            if isinstance(instance, Option):
                instance.update_tte(cur_tte)

        # after ALL instantiating or resetting is done, then execute strategies

        # first, handle arbitrages
        # product_instances["PB1_WITH_PB2"].strategy()

        # second, handle non-arbitrages
        conversions = 0
        for product, instance in product_instances.items():
            if isinstance(instance, Product) and not isinstance(instance, Arbitrage):
                instance.strategy()
                result[product] = instance.orders
                logger.print("Orders for ", product, ": ", instance.orders)
                # NOTE: skip conversions for backtesting
                # if isinstance(instance, Macaron):
                #     conversions = instance.conversions
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData