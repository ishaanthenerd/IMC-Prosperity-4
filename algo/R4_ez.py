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

    def get_rolling_slope(self, window: int) -> float:
        if not hasattr(self, "price_history"):
            self.price_history = []
        
        self.price_history.append(self.mid_price_using_best())
        if len(self.price_history) > window:
            self.price_history.pop(0)

        n = len(self.price_history)
        if n < 2:
            return 0.0
        sum_x = n * (n - 1) / 2
        sum_y = sum(self.price_history)
        sum_xy = sum(i * self.price_history[i] for i in range(n))
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6 
        denom = n * sum_x2 - sum_x**2
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

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

'''
PRODUCT CLASSES
'''

# Configuration for traders to follow per product
WHITELIST = ["Mark 55"]
MAX_PRICE_DIFF = 3

FOLLOW_TRADERS = {
    "VELVETFRUIT_EXTRACT": [],
    "VEV_4000": [],
    "HYDROGEL_PACK": [],
    "VEV_5200": [],
    "VEV_5300": []
}

FOLLOW_TRADERS_INVERTED = {
    "VELVETFRUIT_EXTRACT": ["Mark 55"],
    "VEV_4000": ["Mark 55"],
    "HYDROGEL_PACK": ["Mark 55"],
    "VEV_5200": ["Mark 55"],
    "VEV_5300": ["Mark 55"]
}

USE_SLOPE = False
SLOPE_WINDOW = 200

class BaseCopyTrader(Product):
    def __init__(self, product: str, limit: int, state: TradingState, traders_to_copy: list[str], traders_to_inverse: list[str] = None, window: int = 200, slope_threshold: float = None, price_closeness: int = 5):
        super().__init__(product, limit, state)
        self.traders_to_copy = [t for t in traders_to_copy if t in WHITELIST]
        self.traders_to_inverse = [t for t in (traders_to_inverse if traders_to_inverse else []) if t in WHITELIST]
        self.window = window
        self.slope_threshold = slope_threshold
        self.price_closeness = price_closeness
        # store position history for EV clear if needed
        self.target_position = 0
        
    def fair_val(self):
        return self.mid_price_using_best()

    def get_target_trades(self):
        target_qty = 0
        target_price = self.fair_val()
        
        if self.product in self.market_trades:
            for trade in self.market_trades[self.product]:
                if trade.timestamp == self.timestamp - 100:
                    if trade.buyer in self.traders_to_copy:
                        target_qty += trade.quantity
                        target_price = trade.price
                    if trade.seller in self.traders_to_copy:
                        target_qty -= trade.quantity
                        target_price = trade.price
                        
                    if trade.buyer in self.traders_to_inverse:
                        target_qty -= trade.quantity
                        target_price = trade.price
                    if trade.seller in self.traders_to_inverse:
                        target_qty += trade.quantity
                        target_price = trade.price
        return target_qty, target_price
        
    def strategy(self):
        target_qty, target_price = self.get_target_trades()
        slope = self.get_rolling_slope(self.window)
        
        # Check slope condition if specified
        if USE_SLOPE and self.slope_threshold is not None:
            if target_qty > 0 and slope < self.slope_threshold:
                target_qty = 0
            elif target_qty < 0 and slope > -self.slope_threshold:
                target_qty = 0
                
        # Copy trade
        if target_qty != 0:
            if target_qty > 0:
                # Buy until limit, but only for orders within MAX_PRICE_DIFF of the target price or best bid
                top_price = target_price + MAX_PRICE_DIFF
                for price in sorted(self.order_depth.sell_orders.keys()):
                    if price > top_price:
                        break
                    available = -self.order_depth.sell_orders[price]
                    buy_quantity = min(self.limit_buy_orders(), available)
                    if buy_quantity > 0:
                        self.buy(price, buy_quantity)
            elif target_qty < 0:
                # Sell until limit, but only for orders within MAX_PRICE_DIFF of the target price or best ask
                bottom_price = target_price - MAX_PRICE_DIFF
                for price in sorted(self.order_depth.buy_orders.keys(), reverse=True):
                    if price < bottom_price:
                        break
                    available = self.order_depth.buy_orders[price]
                    sell_quantity = min(self.limit_sell_orders(), available)
                    if sell_quantity > 0:
                        self.sell(price, sell_quantity)

        # 0 EV trade to reduce position back to 0 if position is large
        if abs(self.active_position()) > self.limit * 0.6:
            self.clear(self.fair_val())
            
class VelvetFruitExtract(BaseCopyTrader):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state, 
                         traders_to_copy=FOLLOW_TRADERS.get(product, []), 
                         traders_to_inverse=FOLLOW_TRADERS_INVERTED.get(product, []), 
                         window=200, price_closeness=4)

VEV_CONFIGS = {
    "VEV_4000": {"window": 200, "slope_threshold": 0.001, "price_closeness": 12},
    "VEV_5200": {"window": 200, "slope_threshold": 0.003, "price_closeness": 2},
    "VEV_5300": {"window": 200, "slope_threshold": 0.002, "price_closeness": 2},
}

class VEV(BaseCopyTrader):
    def __init__(self, product, limit, state):
        config = VEV_CONFIGS.get(product, {"window": 200, "slope_threshold": None, "price_closeness": 5})
        super().__init__(product, limit, state, 
                         traders_to_copy=FOLLOW_TRADERS.get(product, []),
                         traders_to_inverse=FOLLOW_TRADERS_INVERTED.get(product, []),
                         window=config.get("window", 200),
                         slope_threshold=config.get("slope_threshold", None),
                         price_closeness=config.get("price_closeness", 5))

class HydrogelPack(BaseCopyTrader):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state, 
                         traders_to_copy=FOLLOW_TRADERS.get(product, []), 
                         traders_to_inverse=FOLLOW_TRADERS_INVERTED.get(product, []), 
                         window=200, price_closeness=10)

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
            product_instances.extend([
                VelvetFruitExtract("VELVETFRUIT_EXTRACT", 250, state),
                HydrogelPack("HYDROGEL_PACK", 250, state),
                VEV("VEV_4000", 600, state),
                VEV("VEV_5200", 600, state),
                VEV("VEV_5300", 600, state)
            ])

            # turn on the trading unit; the products have been populated!
            Trader.turned_on = True
        else:
            # reset ALL the states (can help if multiple product states are entangled)
            for instance in product_instances:
                instance.reset_state(state)

        # after ALL instantiating or resetting is done, then execute strategies
        for instance in product_instances:
            if isinstance(instance, Product):
                instance.strategy()
                result[instance.product] = instance.orders
                logger.print("Orders for ", instance.product, ": ", instance.orders)

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData