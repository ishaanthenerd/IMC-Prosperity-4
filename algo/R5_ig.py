from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
from collections import deque
import string, json, math, statistics
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

statuses = [
    "untraded",
    "default",
    "buy and hold",
    "sell and hold",
    "rolling z",
    "linear fit" # used when seems like buy/hold or sell/hold but too volatile directly
]

product_status = {
    'GALAXY_SOUNDS_BLACK_HOLES': "buy and hold",
    'GALAXY_SOUNDS_DARK_MATTER': "untraded",
    'GALAXY_SOUNDS_PLANETARY_RINGS': "untraded",
    'GALAXY_SOUNDS_SOLAR_FLAMES': "untraded",
    'GALAXY_SOUNDS_SOLAR_WINDS': "untraded",
    'MICROCHIP_CIRCLE': "default",
    'MICROCHIP_OVAL': "sell and hold",
    'MICROCHIP_RECTANGLE': "untraded",
    'MICROCHIP_SQUARE': "default",
    'MICROCHIP_TRIANGLE': "default",
    'OXYGEN_SHAKE_CHOCOLATE': "default",
    'OXYGEN_SHAKE_EVENING_BREATH': "rolling z",
    'OXYGEN_SHAKE_GARLIC': "buy and hold",
    'OXYGEN_SHAKE_MINT': "untraded",
    'OXYGEN_SHAKE_MORNING_BREATH': "untraded",
    'PANEL_1X2': "untraded",
    'PANEL_1X4': "untraded",
    'PANEL_2X2': "untraded",
    'PANEL_2X4': "buy and hold",
    'PANEL_4X4': "untraded",
    'PEBBLES_L': "untraded",
    'PEBBLES_M': "untraded",
    'PEBBLES_S': "sell and hold",
    'PEBBLES_XL': "linear fit",
    'PEBBLES_XS': "sell and hold",
    'ROBOT_DISHES': "buy and hold",
    'ROBOT_IRONING': "linear fit",
    'ROBOT_LAUNDRY': "untraded",
    'ROBOT_MOPPING': "untraded",
    'ROBOT_VACUUMING': "untraded",
    'SLEEP_POD_COTTON': "default",
    'SLEEP_POD_LAMB_WOOL': "untraded",
    'SLEEP_POD_NYLON': "default",
    'SLEEP_POD_POLYESTER': "default",
    'SLEEP_POD_SUEDE': "default",
    'SNACKPACK_CHOCOLATE': "untraded",
    'SNACKPACK_PISTACHIO': "sell and hold",
    'SNACKPACK_RASPBERRY': "untraded",
    'SNACKPACK_STRAWBERRY': "default",
    'SNACKPACK_VANILLA': "untraded",
    'TRANSLATOR_ASTRO_BLACK': "default",
    'TRANSLATOR_ECLIPSE_CHARCOAL': "untraded",
    'TRANSLATOR_GRAPHITE_MIST': "default",
    'TRANSLATOR_SPACE_GRAY': "untraded",
    'TRANSLATOR_VOID_BLUE': "untraded",
    'UV_VISOR_AMBER': "sell and hold",
    'UV_VISOR_MAGENTA': "untraded",
    'UV_VISOR_ORANGE': "default",
    'UV_VISOR_RED': "default",
    'UV_VISOR_YELLOW': "untraded",
}

rolling_z_info = {
    'PANEL_1X4': [1, 100, 9397.5806, 834.0335],
    'PANEL_2X2': [3, 1000, math.nan, math.nan],
    'PANEL_4X4': [1.25, 100, 9878.7193, 457.0376],
    'OXYGEN_SHAKE_EVENING_BREATH': [1, 100, 9367.4741, 220.9269]
}

linear_fit_info = {
    'PEBBLES_XL':    [15776.07294625598,  0.001700265744201406],
    'ROBOT_IRONING': [7532.773349686106, -0.000779171972254632],
}

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
    
    def take_clear_make_balanced(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        self.take(fair_val, edge)
        self.clear(fair_val)
        self.make_balanced(fair_val, edge)

    '''
    SECTION 4 - Non-Implemented Strategies (runtime polymorphism)
    '''
    def fair_val(self):   
        raise NotImplementedError()
        ...

    def strategy(self):
        raise NotImplementedError()
        ...

class DefaultProduct(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)

    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        bid = self.best_bid() + 1
        ask = self.best_ask() - 1
        self.take_clear_make_balanced((bid + ask) / 2, (ask - bid) / 2)

class BuyAndHold(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)

    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        if self.timestamp <= 990000:
            if self.active_position() < self.limit:
                quantity = min(self.max_buy_orders(), -self.order_depth.sell_orders[self.best_ask()])
                self.buy(self.best_ask(), quantity)
        else:
            if self.active_position() > 0:
                self.sell(self.best_bid(), self.active_position())

class SellAndHold(Product):
    def __init__(self, product, limit, state):
        super().__init__(product, limit, state)

    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        if self.timestamp <= 990000:
            if self.active_position() > -self.limit:
                quantity = min(self.max_sell_orders(), self.order_depth.buy_orders[self.best_bid()])
                self.sell(self.best_bid(), quantity)
        else:
            if self.active_position() < 0:
                self.buy(self.best_ask(), -self.active_position())

class LinearFit(Product):
    def __init__(self, product, limit, state, slope: float, intercept: float):
        super().__init__(product, limit, state)
        self.slope = slope
        self.intercept = intercept # assuming the *current* day start is timestamp of 0!!

    def fair_val(self):
        return self.timestamp * self.slope + self.intercept
    
    def strategy(self):
        self.take(self.fair_val())

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

class RollingZProduct(Product):
    def __init__(self, product, limit, state, 
                 z_th: float, window: float, mean: float = math.nan, std: float = math.nan):
        super().__init__(product, limit, state)
        self.rolling = RollingZ(z_th, window, mean, std)

    def fair_val(self):
        return self.mid_price_using_best()
    
    def strategy(self):
        if math.isnan(self.fair_val()):
            return
        self.rolling.add(self.fair_val())
        signal = self.rolling.signal()
        if signal == -1:
            quantity = min(self.max_sell_orders(), self.order_depth.buy_orders[self.best_bid()])
            self.sell(self.best_bid(), quantity)
        elif signal == 1:
            quantity = min(self.max_buy_orders(), -self.order_depth.sell_orders[self.best_ask()])
            self.buy(self.best_ask(), quantity)

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

        if not Trader.turned_on:
            # initiate the products / arbitrages
            for product, status in product_status.items():
                if status not in statuses:
                    raise Exception("ERROR: Invalid status.")
                elif status == "default":
                    product_instances.append(DefaultProduct(product, 10, state))
                elif status == "buy and hold":
                    product_instances.append(BuyAndHold(product, 10, state))
                elif status == "sell and hold":
                    product_instances.append(SellAndHold(product, 10, state))
                elif status == "rolling z":
                    rz_info = rolling_z_info[product]
                    product_instances.append(RollingZProduct(product, 10, state,
                                                             rz_info[0], rz_info[1], rz_info[2], rz_info[3]))
                elif status == "linear fit":
                    lf_info = linear_fit_info[product]
                    product_instances.append(LinearFit(product, 10, state,
                                                       lf_info[0], lf_info[1]))

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