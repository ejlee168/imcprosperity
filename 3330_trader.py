# THIS IS THE MOST STABLE BUILD SO FAR
# Website stats:
# Amethyst:  1674
# Starfruit: 1656
# Total:     3330

import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import numpy as np
import pandas as pd
import math

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
    
    # Helper functions to serialise and deserialise data
    def serialize_to_string(self):
        data = {}
        for attr_name, attr_value in vars(Trader).items():
            if not attr_name.startswith("__") and not callable(attr_value):
                data[attr_name] = attr_value
        serialised_data = jsonpickle.encode(data)
        #logger.print(serialised_data)
        return serialised_data
    
    def deserialize_data(self, string: str):
        data = jsonpickle.decode(string)
        for attr_name, attr_value in data.items():
            setattr(Trader, attr_name, attr_value)
                
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

        if (timestamp == 0): # When the timestamp is 0, set price to 5000
            acceptable_price = default_price

        # When the timestamp is not 0 and the price can be predicted with regression
        elif (predicted_price != -1): 
            acceptable_price = predicted_price
        else: # when the price cannot be predicted with regression, then use average midprice
            acceptable_price = sum(cache)/len(cache)

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

    def get_spread(self, cache):
        y = 0.1
        k = 1.5
        sd = np.std(cache)

        return y*(sd**2) + 2/y * math.log(1 + y/k)
    
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
        # logger.print(f"buy index: {buy_index}, sell index: {sell_index}")
        self.current_positions[product] = cur_pos
        new_orders = sorted_buy_orders[:buy_index] + sorted_sell_orders[:sell_index]
        ##logger.print(f"new orders: {new_orders}")
        
        return new_orders

    # Returns weighted mid price 
    def get_weighted_midprice(self, market_sell_orders: list[(int, int)], market_buy_orders: list[(int, int)]):
        return sum([price*-volume for price, volume in market_sell_orders] + 
                   [price*volume for price, volume in market_buy_orders])/sum([-volume for _, volume in market_sell_orders] +
                                                                               [volume for _, volume in market_buy_orders])

    def compute_amethyst_orders(self, state):
        product = "AMETHYSTS"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        market_sell_orders = list(order_depth.sell_orders.items())
        market_buy_orders = list(order_depth.buy_orders.items())

        best_ask, best_ask_amount = market_sell_orders[0]
        best_bid, best_bid_amount = market_buy_orders[0]

        # Calculate price for amethysts
        acceptable_price = sum(self.amethyst_cache)/len(self.amethyst_cache)
        logger.print("Amethyst acceptable price: ", acceptable_price)

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
        bid_amount = 10

        if (state.position.get(product, 0) > bid_amount):
            bid_amount = math.ceil((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        ask_amount = 10

        if (state.position.get(product, 0) < -ask_amount):
            ask_amount = math.ceil((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        spread = 3
        price = 10000
        undercut = 1

        if not (best_ask <= acceptable_price):
            # Send a buy order (bots will sell to us at this price)
            orders.append(Order(product, min(int(price - spread), best_bid + undercut), bid_amount)) # Want to buy at 9997 -- int(price - spread)

        if not (best_bid >= acceptable_price):
            # Send a sell order (bots will buy from us at this price)
            orders.append(Order(product, max(int(price + spread), best_ask - undercut), -ask_amount)) # SELL should be negative for market making -- int(price + spread)

        return orders
    
    def compute_starfruit_orders(self, state):
        product = "STARFRUIT"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        market_sell_orders = list(order_depth.sell_orders.items())
        market_buy_orders = list(order_depth.buy_orders.items())

        best_ask, best_ask_amount = market_sell_orders[0]
        best_bid, best_bid_amount = market_buy_orders[0]

        # Calculate regression price for starfruit
        acceptable_price_regres = self.get_price_regression('STARFRUIT', 
                                                        self.starfruit_time_cache, 
                                                        self.starfruit_cache, 
                                                        state.timestamp,
                                                        default_price = 5000, # Maybe make it so that it doesn't trade at timestamp 0?
                                                        forecast = 1)
        
        # Calculate price of last 5 starfruit
        acceptable_price_avg = sum(self.starfruit_cache[-5:])/len(self.starfruit_cache[-5:])

        logger.print("Starfruit acceptable regres price: ", acceptable_price_regres, ". avg price: ", acceptable_price_avg)
        logger.print("Starfruit best ask: ", best_ask)
        logger.print("Starfruit best bid: ", best_bid)

        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            ordered_flag = 0
            if best_ask <= acceptable_price_avg: # Buy based on average price
                logger.print(product, " BUY avg", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

            elif best_ask <= acceptable_price_regres and state.timestamp >= 100 * self.get_cache_num("STARFRUIT"): # Buy based on regression price
                logger.print(product, " BUY regres", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))


        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            if best_bid >= acceptable_price_avg:    
                logger.print(product, " SELL avg", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))
            elif best_bid >= acceptable_price_regres and state.timestamp >= 100 * self.get_cache_num("STARFRUIT"):
                logger.print(product, " SELL regres", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))

        # MArket MAKING
        bid_amount = 10

        if (state.position.get(product, 0) > bid_amount):
            bid_amount = int((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        ask_amount = 10

        if (state.position.get(product, 0) < -ask_amount):
            ask_amount = int((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        # spread = 2
        spread = self.get_spread(self.starfruit_cache)
        logger.print(spread)

        price = int(self.get_weighted_midprice(market_sell_orders, market_buy_orders)) # change this to weighted mid price 
        logger.print(price)

        undercut = 1

        if not (best_ask <= acceptable_price_avg) or not (best_ask <= acceptable_price_regres):
            # Send a buy order (bots will sell to us at this price)
            orders.append(Order(product, min(int(price - spread), best_bid + undercut), bid_amount)) # Want to buy -- int(price - spread)

        if not (best_bid >= acceptable_price_avg) or not (best_bid >= acceptable_price_regres):
            # Send a sell order (bots will buy from us at this price)
            orders.append(Order(product, max(int(price + spread), best_ask - undercut), -ask_amount)) # Want to sell --  math.ceil(price + spread) - math.ceil(acceptable_price_avg) - 1)

        return orders
        

    # This method is called at every timestamp -> it handles all the buy and sell orders, and outputs a list of orders to be sent
    def run(self, state: TradingState):
        # update cache only if information is lost
        if state.traderData != "" and self.starfruit_cache == []:
            #logger.print("trader: ", state.traderData)
            self.deserialize_data(state.traderData)
        
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

        # serialise data
        trader_data = self.serialize_to_string() # String value holding Trader state data. Delivered as TradingState.traderData on next execution.

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data