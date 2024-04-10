import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import numpy as np
import pandas as pd

# Logger class is so that https://jmerle.github.io/imc-prosperity-2-visualizer/?/visualizer can be used
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()

class Trader:
    
    POSITION_LIMIT = {'STARFRUIT' : 20, 'AMETHYSTS' : 20}
    current_positions = {'STARFRUIT' : 0, 'AMETHYSTS' : 0}
    
    # Stores cache nums for each product 
    product_cache_num = {"STARFRUIT" : 20, 'AMETHYSTS' : 20}

    # This starfruit_cache stores the last 'starfruit_cache_num' of starfruit midprices
    starfruit_cache = []
    starfruit_time_cache = []

    # Amethyst cache to store amethysts
    amethyst_cache = []
    amethyst_time_cache = []

    # Helper function to store the midprice of a product
    def cache_product(self, product: Symbol, state: TradingState):
        # Get the order depths of product
        order_depth: OrderDepth = state.order_depths[product]

        # Extract the best  ask and bid
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]

        # Add the product midprice and timestamp to relevant cache
        match product:
            case "STARFRUIT":
                self.starfruit_cache.append((best_ask + best_bid)/2)
                self.starfruit_time_cache.append(state.timestamp)
            case "AMETHYSTS":
                self.amethyst_cache.append((best_ask + best_bid)/2)
                self.amethyst_time_cache.append(state.timestamp)

    # Returns the cache num for a product
    def get_cache_num(self, product: Symbol):
        match product:
            case "STARFRUIT":
                return self.product_cache_num['STARFRUIT']
            case "AMETHYSTS":
                return self.product_cache_num['AMETHYSTS']
    
    # Handles caching of a product by removing items from cache before overflow, and appending new midprices to the cache
    def handle_cache(self, product: Symbol, state: TradingState, cache: list[int], time_cache: list[int]):
        cache_num = self.get_cache_num(product)
        
        # Remove prices from cache if there are too many
        if (len(cache) == cache_num): 
            cache.pop(0)
            time_cache.pop(0)
        
        # Store midprice of a product
        self.cache_product(product, state)

    # Calculates regression when given the times and prices, and a timestamp to predict the price of
    def calculate_regression(self, times: list[int], prices: list[int], timestamp: int):
        data = pd.DataFrame({"times": times, "prices": prices})

        # Convert DataFrame columns to NumPy arrays for calculations
        X = np.array(data["times"])
        Y = np.array(data["prices"])

        # Calculate the mean of x and y
        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        # Calculate the slope (m)
        if (np.var(X) == 0):
            return -1 # Return -1 if the variance is 0
        else:
            m = np.sum((X - mean_x) * (Y - mean_y)) / np.sum((X - mean_x) ** 2)

        # Calculate the intercept (c)
        c = mean_y - m * mean_x

        return m * timestamp + c

    # Returns the acceptable_price base on regression
    def get_price_regression(self, product: Symbol, time_cache: list[int], cache: list[int], timestamp, default_price: int, forecast: int):
        # Price is based on regression prediction of next timestamp, otherwise average sum
        predicted_price = self.calculate_regression(time_cache, cache, timestamp + 100 * forecast)
        cache_num = self.get_cache_num(product)
        
        if (timestamp == 0): # When the timestamp is 0, set price to 5000
            acceptable_price = default_price

        # When the timestamp is not 0 and the price can be predicted with regression
        elif (predicted_price != -1): 
            acceptable_price = round(predicted_price, 5)

        else: # when the price cannot be predicted with regression, then use moving average midprice
            acceptable_price = round(sum(cache)/cache_num, 5)

        return acceptable_price
      
    # Returns weighted average
    def get_price_weighted_average(self, 
                                    cache: list[int], 
                                    timestamp: int,
                                    default_price: int):
            if (timestamp == 0):
                return default_price

            cache_size = len(cache)

            sum_of_times = (cache_size*(cache_size + 1)) / 2

            wa = 0
            weights = [0 for _ in range(cache_size)]

            for i in range(len(cache)):
                weights[i] = i/sum_of_times

            wa = sum(weight * price for weight, price in zip(weights, cache))

            return wa

    def adjust_positions(self, orders: List[Order], product: str):
        #logger.print(f"old orders: {orders}")
        limit = self.POSITION_LIMIT[product]
        #logger.print(f"limit: {limit}")
        cur_pos = self.current_positions[product]
        #logger.print(f"current position: {cur_pos}")
        
        # separate and sort orders by best price
        buy_orders = [order for order in orders if order.quantity > 0]
        sell_orders = [order for order in orders if order.quantity < 0]
        sorted_buy_orders = sorted(buy_orders, key=lambda x: x.price)
        sorted_sell_orders = sorted(sell_orders, key=lambda x: x.price, reverse=True)
        #logger.print(f"sorted buy orders: {sorted_buy_orders}")
        #logger.print(f"sorted sell orders: {sorted_sell_orders}")

        # iterate until position limit is reached
        buy_index = 0
        sell_index = 0

        while buy_index < len(buy_orders) and sell_index < len(sell_orders):
            #logger.print(f"buy index: {buy_index}, sell index: {sell_index}")
            if buy_index < len(buy_orders):
                #logger.print(f"before buy: {cur_pos}")
                if sorted_buy_orders[buy_index].quantity + cur_pos > limit:
                    diff = limit - cur_pos
                    sorted_buy_orders[buy_index].quantity = diff 
                cur_pos += sorted_buy_orders[buy_index].quantity
                buy_index += 1
                #logger.print(f"after buy: {cur_pos}")
            if sell_index < len(sell_orders):
                #logger.print(f"before sell: {cur_pos}")
                if sorted_sell_orders[sell_index].quantity + cur_pos < -limit:
                    diff = limit - cur_pos
                    sorted_sell_orders[sell_index].quantity = diff
                cur_pos += sorted_sell_orders[sell_index].quantity
                sell_index += 1
                #logger.print(f"after sell: {cur_pos}")
            
        while buy_index < len(buy_orders):
            #logger.print(f"before buy: {cur_pos}")
            if sorted_buy_orders[buy_index].quantity + cur_pos > limit:
                    diff = limit - cur_pos
                    sorted_buy_orders[buy_index].quantity = diff 
            cur_pos += sorted_buy_orders[buy_index].quantity
            buy_index += 1
            #logger.print(f"after buy: {cur_pos}")
            
        while sell_index < len(sell_orders):
            #logger.print(f"before sell: {cur_pos}")
            if sorted_sell_orders[sell_index].quantity + cur_pos < -limit:
                    diff = limit - cur_pos
                    sorted_sell_orders[sell_index].quantity = diff 
            cur_pos += sorted_sell_orders[sell_index].quantity
            sell_index += 1
            #logger.print(f"after sell: {cur_pos}")

        # update position and return adjusted orders
        #logger.print(f"final position: {cur_pos}")
        logger.print(f"buy index: {buy_index}, sell index: {sell_index}")
        self.current_positions[product] = cur_pos
        new_orders = sorted_buy_orders[:buy_index] + sorted_sell_orders[:sell_index]
        ##logger.print(f"new orders: {new_orders}")
        
        return new_orders

    def compute_amethyst_orders(self, state):
        product = "AMETHYSTS"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        # Calculate price for amethysts
        acceptable_price = sum(self.amethyst_cache)/len(self.amethyst_cache)
        logger.print("Amethyst acceptable price: ", acceptable_price)
        logger.print("Amethyst best ask: ", best_ask)
        logger.print("Amethyst best bid: ", best_bid)

        # Market TAKING:
        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            if best_ask <= acceptable_price:
                logger.print(product, " BUY", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            if best_bid >= acceptable_price:
                logger.print(product, " SELL", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))

        # Market MAKING
        amount = 10
        if (state.position.get("AMETHYSTS", 0) + amount > self.POSITION_LIMIT["AMETHYSTS"]) or \
            (state.position.get("AMETHYSTS", 0) - amount < -self.POSITION_LIMIT["AMETHYSTS"]):
            amount = self.POSITION_LIMIT["AMETHYSTS"] - abs(state.position.get("AMETHYSTS", 0))

        spread = 3
        price = int(self.amethyst_cache[-1]) # change this to weighted mid price, at the moment it is just current midprice
        # price = 10000

        if best_ask > acceptable_price:
            # Send a buy order
            orders.append(Order("AMETHYSTS", price - spread, amount)) # Want to buy at 9996
        
        if best_bid < acceptable_price:
            # Send a sell order
            orders.append(Order("AMETHYSTS", price + spread, -amount)) # SELL should be negative for market making

        return orders
    
    def compute_starfruit_orders(self, state):
        product = "STARFRUIT"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        # Calculate price for starfruit
        acceptable_price = self.get_price_regression('STARFRUIT', 
                                                        self.starfruit_time_cache, 
                                                        self.starfruit_cache, 
                                                        state.timestamp,
                                                        default_price = 5000,
                                                        forecast = 0)
        logger.print("Starfruit acceptable price: ", acceptable_price)
        logger.print("Starfruit best ask: ", best_ask)
        logger.print("Starfruit best bid: ", best_bid)

        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            if best_ask <= acceptable_price:
                logger.print(product, " BUY", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            if best_bid >= acceptable_price:
                logger.print(product, " SELL", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))
        
        return orders

    # This method is called at every timestamp -> it handles all the buy and sell orders, and outputs a list of orders to be sent
    def run(self, state: TradingState):

        # Update positions
        for product in self.current_positions:
            if product in state.position:
                self.current_positions[product] = state.position[product]
            
        # Dictionary that will end up storing all the orders of each product
        result = {}
        
        # Handle cache for starfruit and amethyst
        self.handle_cache('STARFRUIT', state, self.starfruit_cache, self.starfruit_time_cache)
        self.handle_cache('AMETHYSTS', state, self.amethyst_cache, self.amethyst_time_cache)

        # Get amethyst orders
        amethyst_orders = self.compute_amethyst_orders(state)
        result["AMETHYSTS"] = self.adjust_positions(amethyst_orders, "AMETHYSTS")

        # Get starfruit orders
        starfruit_orders = self.compute_starfruit_orders(state)
        result["STARFRUIT"] = self.adjust_positions(starfruit_orders, "STARFRUIT")

        trader_data = "" # String value holding Trader state data. Delivered as TradingState.traderData on next execution.

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data