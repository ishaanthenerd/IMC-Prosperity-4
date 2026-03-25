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
        self.order_depth = state.order_depths[product]
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
        self.order_depth = new_state.order_depths[self.product]
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
    def mid_price(self):
        return self.mid_price_using_worst() # default is use worst
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
        self.buy(buy_price, self.limit_buy_orders())
        self.sell(sell_price, self.limit_sell_orders())
    
    # market-making under the condition of starting at [fv - edge, fv + edge] and expanding outwards
    def market_make_undercut(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1
        self.market_make(mm_buy, mm_sell)
    
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
    def __init__(self, z_th: float, window: int, fixed_mean: float = math.nan):
        self.premiums = np.array([], dtype = float)
        self.z_th = z_th
        self.window = window
        self.fixed_mean = fixed_mean

    def add(self, new_val: float):
        self.premiums = np.append(self.premiums, new_val)
        if len(self.premiums) > self.window:
            np.delete(self.premiums, 0)
        assert(len(self.premiums) != 0)

    def mean(self):
        return self.fixed_mean if not math.isnan(self.fixed_mean) else np.mean(self.premiums)

    def std(self):
        return np.std(self.premiums)
    
    def most_recent(self):
        return self.premiums[-1]

    # 1 means buy, 0 means do nothing, -1 means sell (sign indicates position direction)
    def signal(self):
        z_score = 0
        if math.isnan(self.fixed_mean):
            z_score = (self.premiums[-1] - np.mean(self.premiums)) / np.std(self.premiums)
        else:
            z_score = (self.premiums[-1] - self.fixed_mean) / np.std(self.premiums)

        if z_score < -self.z_th:
            return 1
        elif z_score > self.z_th:
            return -1
        else:
            return 0

# START OF NEW CLASSES

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
        self.time_to_expiry = new_tte
        self.underlying_bought = 0
        self.underlying_sold = 0

    # FLIP THIS IF NEEDED!!
    def will_delta_hedge(self):
        return True

    def fair_val(self):
        spot = self.underlying_prices.most_recent()
        avg_iv = self.ivs.mean()
        if self.is_call:
            return BlackScholes.black_scholes_call(spot, self.strike, self.tte, avg_iv)
        else:
            return BlackScholes.black_scholes_put(spot, self.strike, self.tte, avg_iv)
    
    def strategy(self):
        # add data to three RollingZ classes
        self.underlying_prices.add(self.underlying.mid_price())
        cur_mid = self.mid_price()
        if self.is_call:
            iv = BlackScholes.implied_volatility_call(cur_mid, self.underlying_prices.most_recent(), self.strike, self.tte)
            delta = BlackScholes.delta_call(self.underlying_prices.most_recent(), self.strike, self.tte, iv)
        else:
            iv = BlackScholes.implied_volatility_put(cur_mid, self.underlying_prices.most_recent(), self.strike, self.tte)
            delta = BlackScholes.delta_put(self.underlying_prices.most_recent(), self.strike, self.tte, iv)
        self.ivs.add(iv)
        self.deltas.add(delta)

        # compute the fair value
        true_value = self.fair_val()
        spot = self.underlying_prices.most_recent()
        avg_iv = self.ivs.mean()
        std_iv = self.ivs.std()
        if self.is_call:
            low_value = BlackScholes.black_scholes_call(spot, self.strike, self.tte, avg_iv - std_iv)
            high_value = BlackScholes.black_scholes_call(spot, self.strike, self.tte, avg_iv + std_iv)
        else:
            low_value = BlackScholes.black_scholes_put(spot, self.strike, self.tte, avg_iv - std_iv)
            high_value = BlackScholes.black_scholes_put(spot, self.strike, self.tte, avg_iv + std_iv)
        
        # if not volatile enough, just rebalance position
        if (high_value - low_value) < 1.0:
            if self.active_position() < 0:
                self.buy(self.fair_val(), -self.active_position())
            elif self.active_position() > 0:
                self.sell(self.fair_val(), self.active_position())

        # now make our market
        eps = 0.1
        bid_val = int(floor(true_value + eps))
        ask_val = int(ceil(true_value - eps))

        # market make BUT keep track of sizings
        bid_size = self.limit - self.active_position()
        ask_size = self.limit + self.active_position()

        total_bought = 0
        total_sold = 0
        true_value = int(true_value)
        max_spread = floor(true_value * 0.03)

        # crossing with best bid
        for market_bid, market_amount in self.order_depth.buy_orders.items():
            if ask_val < market_bid:
                # eat their market then take it over
                eat_order_size = abs(min(ask_size, abs(market_amount)))
                total_sold += eat_order_size
                self.sell(self.best_bid(), eat_order_size)
                
                # place ask below best ask
                if not math.isnan(self.best_ask()): 
                    ask_val = self.best_ask() - 1
                else:
                    ask_val = int(math.ceil(ask_val + max_spread))
                
                # place bid at maximum dist from fair value
                bid_val = int(math.floor(max(true_value - max_spread, bid_val)))
        
        # crossing with best ask
        for market_ask, market_amount in self.order_depth.sell_orders.items():
            if bid_val > market_ask:
                # eat their market then take it over
                eat_order_size = abs(min(bid_size, abs(market_amount)))
                self.buy(market_ask, eat_order_size)
                total_bought += eat_order_size
                    
                # place bid above best bid
                if not math.isnan(self.best_bid()):
                    bid_val = self.best_bid() + 1
                else:
                    bid_val = int(math.floor(true_value - max_spread))

                # place ask at maximum dist from fair value
                ask_val = int(math.ceil(max(true_value + max_spread, ask_val)))

        if self.will_delta_hedge():
            # calculate the amount of delta bought and sold, then rebalance delta
            if total_bought > 0:
                bought_delta = self.deltas.mean() * total_bought
                self.sell_underlying(int(bought_delta))
            if total_sold > 0:
                sold_delta = self.deltas.mean() * total_sold
                self.buy_underlying(int(sold_delta))

        # market make to balance position on options back to zero
        bid_size = max(self.limit - self.active_position() - total_bought, 0)
        ask_size = max(self.active_position() + self.limit - total_sold, 0)

        if bid_val == ask_val:
            if self.max_buy_orders() > self.max_sell_orders():
                self.buy(bid_val, bid_size)
            elif self.max_sell_orders() > self.max_buy_orders():
                self.sell(ask_val, ask_size)
        else:
            self.buy(bid_val, bid_size)
            self.sell(ask_val, ask_size)

    def buy_underlying(self, quantity):
        try:
            underlying_value = self.underlying_prices.most_recent()
            underlying_value = int(ceil(underlying_value))
            cur_pos = self.underlying.active_position()
            trade_size = min(self.underlying.limit - cur_pos - self.underlying_bought, quantity)
            self.underlying_bought += trade_size
            self.underlying.buy(underlying_value, trade_size)
        except:
            return

    def sell_underlying(self, quantity):
        try:
            underlying_value = self.underlying_prices.most_recent()
            underlying_value = int(floor(underlying_value))
            cur_pos = self.underlying.active_position()
            trade_size = min(self.underlying.limit + cur_pos - self.underlying_sold, quantity)
            self.underlying_sold += trade_size
            self.underlying.sell(underlying_value, trade_size)
        except:
            pass

# END OF NEW CLASSES

'''
PRODUCT CLASSES
'''

# PROSPERITY 3

# it's time to make the ROCK
class Rock(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)
    
    def fair_val(self):
        return self.mid_price()
    
    def strategy(self):
        pass
        # self.market_take(self.fair_val(), 0.5)

class RockVoucher(Option):
    def __init__(self, symbol, limit, state, is_call, strike, tte, underlying, underlying_z_th, underlying_window, iv_z_th, iv_window, delta_z_th, delta_window):
        super().__init__(symbol, limit, state, is_call, strike, tte, underlying, underlying_z_th, underlying_window, iv_z_th, iv_window, delta_z_th, delta_window)
    
    def fair_val(self):
        return super().fair_val()
    
    def strategy(self):
        super().strategy()

# PROSPERITY 4

'''
class Emerald(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        fv = self.fair_val()
        th = 1 # tighest market possible
        self.market_take(fv, th)
        self.market_make_undercut(fv, th)

class Tomato(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return self.mid_price()
    
    def strategy(self):
        fv = self.fair_val()
        th = 5 # experiment!
        buy_orders = self.order_depth.buy_orders
        sell_orders = self.order_depth.sell_orders

        if len(sell_orders) != 0 and len(buy_orders) != 0:
            # initial market
            buy_price = fv - th
            sell_price = fv + th

            # someone else's market is wider - copy them!
            if not math.isnan(self.best_bid()):
                buy_price = min(buy_price, self.best_bid())
            if not math.isnan(self.best_ask()):
                sell_price = max(sell_price, self.best_ask())

            # don't trade opposite of our position side (risk aversion measure)
            if not (self.position > 0 and float(buy_price) >= fv):
                self.buy(buy_price, self.max_buy_orders())
            if not (self.position < 0 and float(sell_price) <= fv):
                self.sell(sell_price, self.max_sell_orders())
'''

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
        days_left = 5
        cur_tte = (days_left / 365) - (state.timestamp / 365e6)

        if not Trader.turned_on:
            # initiate the products / arbitrages

            # PROSPERITY 3
            product_instances["VOLCANIC_ROCK"] = Rock("VOLCANIC_ROCK", 400, state)
            product_instances["VOLCANIC_ROCK_VOUCHER_9500"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_9500", 200, state, True, 9500, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 10, 20, 10, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_9750"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_9750", 200, state, True, 9750, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 10, 20, 10, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10000"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10000", 200, state, True, 10000, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 10, 20, 10, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10250"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10250", 200, state, True, 10250, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 10, 20, 10, 20, 10)
            product_instances["VOLCANIC_ROCK_VOUCHER_10500"] = RockVoucher("VOLCANIC_ROCK_VOUCHER_10500", 200, state, True, 10500, cur_tte, product_instances["VOLCANIC_ROCK"], 20, 10, 20, 10, 20, 10)

            # PROSPERITY 4
            '''
            product_instances["EMERALDS"] = Emerald("EMERALDS", 80, state)
            product_instances["TOMATOES"] = Tomato("TOMATOES", 80, state)
            '''
            
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
        for product, instance in product_instances.items():
            if isinstance(instance, Product):
                instance.strategy()
                result[product] = instance.orders
                logger.print("Orders for ", product, ": ", instance.orders)

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData