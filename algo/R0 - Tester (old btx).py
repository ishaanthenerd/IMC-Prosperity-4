from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
import string, json, math

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

class Product():
    def __init__(self, product: str, limit: int, state: TradingState):
        self.state = state
        self.traderData = state.traderData
        self.timestamp = state.timestamp
        self.listings = state.listings
        self.order_depth = state.order_depths[product]
        self.own_trades = state.own_trades
        self.market_trades = state.market_trades
        self.position = state.position
        self.observations = state.observations
        self.product = product
        self.limit = limit
        self.position = state.position[product] if product in state.position else 0
        self.nsell = 0
        self.nbuy = 0
        self.hist_mid: List[float] = []
        self.hist_mm_mid: List[float]= []
        self.hist_sum = 0
        self.hist_sum_squared = 0
        self.hist_mean = 0
        self.hist_std = 0
        self.init_hist_mid()
        self.orders: List[Order] = []
        self.window = 10

    def orderbook_buy_size(self):
        return sum(self.order_depth.buy_orders.values())
    def orderbook_sell_size(self):
        return -sum(self.order_depth.sell_orders.values())
    
    def limit_buy_orders(self):
        return self.limit - self.position - self.nbuy 
    def limit_sell_orders(self):
        return self.limit + self.position - self.nsell
    
    def max_buy_orders(self):
        return min(self.limit_buy_orders(), self.orderbook_sell_size())
    def max_sell_orders(self):
        return min(self.limit_sell_orders(), self.orderbook_buy_size())
    
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
    
    def full_buy(self, quantity: int):
        """Buy the orderbook until the quantity of shares are bought. Limited by max_buy_orders."""
        q = quantity
        for price, volume in self.order_depth.sell_orders.items():
            if volume < 0:
                buy_vol = min(q, min(self.limit_buy_orders(), -volume))
                self.buy(price, buy_vol)
                q -= buy_vol
                if q <= 0 or self.limit_buy_orders() <= 0:
                    break
    def full_sell(self, quantity: int):
        """Sell the orderbook until the quantity of shares are sold. Limited by max_sell_orders."""
        q = quantity
        for price, volume in self.order_depth.buy_orders.items():
            if volume > 0:
                sell_vol = min(q, min(self.limit_sell_orders(), volume))
                self.sell(price, sell_vol)
                q -= sell_vol
                if q <= 0 or self.limit_sell_orders() <= 0:
                    break

    def cancel_orders(self):
        """Cancel all orders."""
        self.orders = []
        self.nsell = 0
        self.nbuy = 0   
    def cancel_buy_orders(self):
        """Cancel all buy orders."""
        self.orders = [order for order in self.orders if order.quantity < 0]
        self.nsell = 0
        self.nbuy = 0
    def cancel_sell_orders(self):
        """Cancel all sell orders."""
        self.orders = [order for order in self.orders if order.quantity > 0]
        self.nsell = 0
        self.nbuy = 0

    def active_position(self):
        return self.position + self.nbuy - self.nsell   
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

    def mid_price(self):
        return (self.best_bid() + self.best_ask()) / 2
        
    def market_take(self, fair_val: float, edge: float = 0):

        bid_val = fair_val - edge
        ask_val = fair_val + edge

        for bid_price, bid_vol in self.order_depth.buy_orders.items():
            if bid_price > bid_val or (bid_price == bid_val and self.active_position() > 0): 
                sell_vol = min(self.max_sell_orders(), bid_vol)
                if bid_price == bid_val:
                    sell_vol = min(sell_vol, self.active_position())
                if sell_vol > 0:
                    self.sell(bid_price, sell_vol)

        for ask_price, ask_vol in self.order_depth.sell_orders.items():
            ask_vol *= -1
            if ask_price < ask_val or (ask_price == ask_val and self.active_position() < 0):
                buy_vol = min(self.max_buy_orders(), ask_vol)
                if ask_price == ask_val:
                    buy_vol = min(buy_vol, -self.active_position())
                if buy_vol > 0:
                    self.buy(ask_price, buy_vol)

    def market_make(
        self,
        buy_price: int,
        sell_price: int
    ):
        self.buy(buy_price, self.limit_buy_orders())
        self.sell(sell_price, self.limit_sell_orders())
    
    def market_make_undercut(
        self,
        fair_val: int,
        edge: float = 0,
    ):
        mm_buy = max([bid for bid in self.order_depth.buy_orders.keys() if bid < fair_val - edge], default=fair_val - edge - 1) + 1
        mm_sell = min([ask for ask in self.order_depth.sell_orders.keys() if ask > fair_val + edge], default=fair_val + edge + 1) - 1

        return self.market_make(mm_buy, mm_sell)
    
    def max_vol_mid(self):
        mm_bid_price, mm_bid_qty = max(self.order_depth.buy_orders.items(), key = lambda x: x[1])
        mm_ask_price, mm_ask_qty = max(self.order_depth.sell_orders.items(), key = lambda x: x[1])
        return (mm_bid_price + mm_ask_price)/2
    
    def init_hist_mid(self):
        prods = self.traderData.split("\n")
        ind = -1
        for i in range(len(prods)):
            arr = prods[i].split(" ")
            if arr[0] == self.product:
                ind = i
                break
        
        if ind >= 0:
            for price in prods[ind].split(" ")[3:]:
                self.hist_mid.append(float(price))
            for mm_price in prods[ind + 1].split(" ")[3:]:
                self.hist_mm_mid.append(float(mm_price))

            iter = self.state.timestamp / 100

            self.hist_sum = float(prods[ind].split(" ")[1])
            self.hist_sum_squared = float(prods[ind].split(" ")[2])
            self.hist_mean = self.hist_sum / iter
            self.hist_std = math.sqrt(self.hist_sum_squared / iter - self.hist_mean ** 2)

    def return_mids(self):
        """Converts self.hist_mid and self.hist_mm_mid into string form for TraderData.
        String form: [PRODUCT] [sum] [sum^2] [mp1] [mp2] ... [mp10]
        [PRODUCT] [sum] [sum^2] [mm_mp1] [mm_mp2] ... [mm_mp10]"""
        
        self.hist_mid.append(self.mid_price())
        self.hist_mm_mid.append(self.max_vol_mid())
        self.hist_mid = self.hist_mid[-self.window:]
        self.hist_mm_mid = self.hist_mm_mid[-self.window:]
        res = self.product
        res += " " + str(self.mid_price() + self.hist_sum)
        res += " " + str(self.mid_price() ** 2 + self.hist_sum_squared)
        for price in self.hist_mid:
            res = res + " " + str(price)
        res += "\n" + self.product
        for price in self.hist_mm_mid:
            res = res + " " + str(price)
        return res

    def hist_mid_make(self, mm_bot: bool=False):
        hm = self.hist_mid
        if mm_bot:
            hm = self.hist_mm_mid
        
        if len(hm) > 0:
            value = sum(hm) / len(hm)
        else:
            if mm_bot:
                value = self.max_vol_mid()
            else:
                value = self.mid_price()
        offset = max(1, (self.best_ask() - self.best_bid()) // 2)
        buy_price = round(value - offset)
        sell_price = round(value + offset)
        order_size = min(self.limit_buy_orders(), self.limit_sell_orders())
        self.buy(buy_price, order_size)
        self.sell(sell_price, order_size)

    def buy_one(self):
        """Utility function to buy one share at the start and test PnL."""
        if self.position == 0:
            if len(self.order_depth.sell_orders) > 0:
                self.buy(self.best_ask(), 1)

    def fair_val(self): # Children inherit the default fair_val   
        mid = self.max_vol_mid()
        prev_mid = mid
        if len(self.hist_mm_mid) > 0:
            prev_mid = self.hist_mm_mid[-1]

        val = mid * 0.9 + prev_mid * 0.1
        return val

    def strategy(self): # RUNTIME POLYMORPHISM BTW
        raise NotImplementedError()
        ...

    def execute(self, blank: bool=False): 
        if blank:
            return [], ""
        self.strategy()
    
    def getData(self):
        return self.return_mids()

'''
Individual products start here!
'''

class Emerald(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return 10000
    
    def strategy(self):
        fv = self.fair_val()
        self.market_take(fv)
        self.market_make_undercut(fv, 1)

class Tomato(Product):
    def __init__(self, symbol: str, limit: int, state: TradingState):
        super().__init__(symbol, limit, state)

    def fair_val(self):
        return self.mid_price()
    
    def strategy(self):
        fv = self.fair_val()
        self.market_make_undercut(fv, 5)

'''
Now create all of the products...
'''

def create_products(state: TradingState):
    products = {}
    products["EMERALDS"] = Emerald("EMERALDS", 80, state)
    products["TOMATOES"] = Tomato("TOMATOES", 80, state)
    return products

'''
And trade them!
'''

class Trader:

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        # log("traderData: " + state.traderData, 2)
        # log("Observations: " + str(state.observations), 2)

        result = {}

        traderData = ""
        product_instances = create_products(state)

        for product, instance in product_instances.items():
            if product in ["EMERALDS", "TOMATOES"]:
                instance.execute()

        for product, instance in product_instances.items():   
            # check if instance is instance of Product
            if isinstance(instance, Product):
                traderData += instance.getData() + "\n"
                result[product] = instance.orders
                logger.print("Orders for ", product, ": ", instance.orders)

        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData