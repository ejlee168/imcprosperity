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
    
    POSITION_LIMIT = {'STARFRUIT' : 20, 'AMETHYSTS' : 20, 'ORCHIDS': 100}
    current_positions = {'STARFRUIT' : 0, 'AMETHYSTS' : 0, 'ORCHIDS': 0}
    
    # Stores cache nums for each product 
    product_cache_num = {"STARFRUIT" : 20, 'AMETHYSTS' : 20, 'ORCHIDS' : 20}
    # signal_cache_num = {"SUNLIGHT" : 100, "HUMIDITY" : 5}

    # This starfruit_cache stores the last 'starfruit_cache_num' of starfruit midprices
    starfruit_cache = []
    starfruit_time_cache = []

    # Amethyst cache to store amethysts
    amethyst_cache = []
    amethyst_time_cache = []

    # orchid cache
    orchid_cache = []
    orchid_time_cache = []

    observations_cache = {"IMPORT_TARIFF": [], "EXPORT_TARIFF": []}

    # sunlight_cache = {0: 12312312, 1: 21312903129}
    sunlight_cache = {}

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
    def cache_product(self, product: Symbol, state: TradingState, cache: list[int], time_cache: list[int]):
        # Get the order depths of product
        order_depth: OrderDepth = state.order_depths[product]

        # Extract the best  ask and bid
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]

        # Add the product midprice and timestamp to relevant cache
        cache.append((best_ask + best_bid)/2)
        time_cache.append(state.timestamp)

    # Handles caching of a product by removing items from cache before overflow, and appending new midprices to the cache
    def handle_cache(self, product: Symbol, state: TradingState, cache: list[int], time_cache: list[int]):

        if (product == "STARFRUIT" or product == "AMETHYSTS" or product == "ORCHIDS"):
            cache_num = self.product_cache_num[product]
            
            # Remove prices from cache if there are too many
            if (len(cache) == cache_num): 
                cache.pop(0)
                time_cache.pop(0)
            
            # Store midprice of a product
            self.cache_product(product, state, cache, time_cache)

    def handle_cache_sunlight(self, state: TradingState):
        sunlight = state.observations.conversionObservations['ORCHIDS'].sunlight
        current_sum = self.sunlight_cache.get(state.timestamp//83300,0)
        self.sunlight_cache[state.timestamp//83300] = current_sum + sunlight

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
    def get_price_regression(self, time_cache: list[int], cache: list[int], timestamp, default_price: int, forecast: int):
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
        acceptable_price = 10000
        logger.print(f"{product} acceptable price: {acceptable_price}")

        # Market TAKING:
        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            if best_ask <= acceptable_price:
                amount = min(-best_ask_amount, self.POSITION_LIMIT[product] - state.position.get(product, 0))

                # Market take
                orders.append(Order(product, best_ask, amount))
                logger.print(product, " BUY", str(amount) + "x", best_ask)

                # Now market make for 2 or until pos limit at an undercut
                amount = min(20, self.POSITION_LIMIT[product] - (state.position.get(product, 0) + amount))
                
                spread = 3
                orders.append(Order(product, min(int(10000 - spread), best_ask + 1), amount))
                logger.print(product, " BUY undercut", str(amount) + "x", best_ask)

        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            if best_bid >= acceptable_price:
                amount = max(-best_bid_amount, -self.POSITION_LIMIT[product] - state.position.get(product, 0))

                # Market take
                orders.append(Order(product, best_bid, amount))
                logger.print(product, " SELL", str(amount) + "x", best_bid)

                # Now market make for 2 or untili pos limit at an undercut
                amount = max(-20, -self.POSITION_LIMIT[product] - (state.position.get(product, 0) + amount))
                
                spread = 3
                orders.append(Order(product, max(int(10000 + spread), best_bid - 1), amount))
                logger.print(product, " SELL", str(amount) + "x", best_bid)
                

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
        acceptable_price_regres = self.get_price_regression(self.starfruit_time_cache, self.starfruit_cache, state.timestamp,
                                                            default_price = 5000, 
                                                            forecast = 1)
        
         # Calculate price of last 5 starfruit
        acceptable_price_avg = sum(self.starfruit_cache[-5:])/len(self.starfruit_cache[-5:])

        logger.print("Starfruit acceptable regres price: ", acceptable_price_regres, ". avg price: ", acceptable_price_avg)
        logger.print("Starfruit best ask: ", best_ask)
        logger.print("Starfruit best bid: ", best_bid)

        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            # Buy based on average price
            if best_ask <= acceptable_price_avg: 
                logger.print(product, " BUY avg", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

             # Buy based on regression price
            elif best_ask <= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]:
                logger.print(product, " BUY regres", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            # Sell based on average price
            if best_bid >= acceptable_price_avg:    
                logger.print(product, " SELL avg", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))

            # Sell based on regression price
            elif best_bid >= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]:
                logger.print(product, " SELL regres", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))

        # MArket MAKING
        bid_amount = 10

        if (state.position.get(product, 0) > bid_amount):
            bid_amount = int((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        ask_amount = 10

        if (state.position.get(product, 0) < -ask_amount):
            ask_amount = int((self.POSITION_LIMIT[product] - abs(state.position.get(product, 0))) * 0.5)

        spread = self.get_spread(self.starfruit_cache)
        price = self.get_weighted_midprice(market_sell_orders, market_buy_orders)
        undercut = 1

        if not (best_ask <= acceptable_price_avg) or not (best_ask <= acceptable_price_regres):
            # Send a buy order (bots will sell to us at this price and we are looking to buy)
            orders.append(Order(product, min(int(price - spread), best_bid + undercut), bid_amount)) 

        if not (best_bid >= acceptable_price_avg) or not (best_bid >= acceptable_price_regres):
            # Send a sell order (bots will buy from us at this price and we are looking to sell)
            orders.append(Order(product, max(int(price + spread), best_ask - undercut), -ask_amount)) 

        return orders

    def get_humidity_change(self, observations: Observation):
            humidity = observations.conversionObservations['ORCHIDS'].humidity
            
            percentage_change = 0

            # return 'no change' if humidity is in optimal range
            if (60.0 < humidity and humidity < 80.0):
                return percentage_change
            
            elif (humidity < 60.0):
                steps = (60.0 - humidity) / 5
                return percentage_change - steps * 2

            elif (80.0 < humidity):
                steps = (humidity - 80) / 5
                return percentage_change - steps * 2

    def get_sunlight_change(self, state: TradingState):
        current_hour = state.timestamp//83300

        sum = 0
        for hour in range(current_hour + 1):
            sum += self.sunlight_cache[hour]

        percentage_change = -4

        if sum < 833 * 7 * 2500: # 7 hrs of sunlight
            return percentage_change
        else:
            return 0

    # calculate orchid storage fees 
    def get_current_storage_fee(self):
        storage_fee = 0.1
        return self.current_positions['ORCHIDS'] * storage_fee

    # Returns cost of orchid trade
    def get_orchid_conversion_cost(self, observations: Observation, conversions: int):
        cost = 0
        if conversions > 0:
            # add cost of orchids (add conversion logic)
            ask_price = observations.conversionObservations['ORCHIDS'].askPrice
            cost += ask_price
            # add tariff, transport and storage
            cost += observations.conversionObservations['ORCHIDS'].importTariff
            # cost += self.get_orchid_storage_cost('buy', conversions) # is this right
        elif conversions < 0:
            # add cost of orchids (add conversion logic)
            bid_price = observations.conversionObservations['ORCHIDS'].bidPrice
            cost += bid_price
            # add tariff, transport and storage
            cost += observations.conversionObservations['ORCHIDS'].exportTariff
            # cost += self.get_orchid_storage_cost('sell', -conversions) # is this right

        cost *= conversions
        cost += observations.conversionObservations['ORCHIDS'].transportFees
        return cost

    # We need some function that calculates orchid prices based on sunlight and humidity
    def get_orchid_price(self, state: TradingState):
        # production_percent_change = self.get_humidity_change(state.observations) + self.get_sunlight_change(state)
        # logger.print("Change ", production_percent_change)

        sunlight = state.observations.conversionObservations['ORCHIDS'].sunlight
        humidity = state.observations.conversionObservations['ORCHIDS'].humidity

        c = 693.3200659
        sunlight_coef = 0.040136744
        humidity_coef = 3.779203831
        regression_price = c + sunlight_coef*sunlight + humidity_coef*humidity

        return regression_price

    def computer_orchid_orders(self, state: TradingState):
        product = "ORCHIDS"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []

        market_sell_orders = list(order_depth.sell_orders.items())
        market_buy_orders = list(order_depth.buy_orders.items())

        best_ask, best_ask_amount = market_sell_orders[0]
        best_bid, best_bid_amount = market_buy_orders[0]

        conversions = 0

        bid_amount = 100

        ask_amount = 100

        price = (best_ask + best_bid)/2

        # buying
        spread = 3
        avgExport = sum(self.observations_cache["EXPORT_TARIFF"])/len(self.observations_cache["EXPORT_TARIFF"])
        offer_price = state.observations.conversionObservations['ORCHIDS'].bidPrice - avgExport - spread
        orders.append(Order(product, int(offer_price), bid_amount))

        # selling
        spread = 2 # spread of 2 is 78k, spread 1 = 28k, spread 3 = 76k
        avgImport = sum(self.observations_cache["IMPORT_TARIFF"])/len(self.observations_cache["IMPORT_TARIFF"])
        logger.print(f"import {avgImport}")
        offer_price = state.observations.conversionObservations['ORCHIDS'].askPrice + avgImport + spread # state.observations.conversionObservations['ORCHIDS'].importTariff
        orders.append(Order(product, math.ceil(offer_price), -ask_amount))

        conversions = -self.current_positions[product]

        return orders, conversions


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

        if (len(self.observations_cache["IMPORT_TARIFF"]) == 20): 
                self.observations_cache["IMPORT_TARIFF"].pop(0)

        observation = state.observations.conversionObservations['ORCHIDS']

        self.observations_cache["IMPORT_TARIFF"].append(observation.importTariff)

        if (len(self.observations_cache["EXPORT_TARIFF"]) == 20): 
                self.observations_cache["EXPORT_TARIFF"].pop(0)

        self.observations_cache["EXPORT_TARIFF"].append(observation.exportTariff)
        

        # Get amethyst orders
        amethyst_orders = self.compute_amethyst_orders(state)
        result["AMETHYSTS"] = amethyst_orders

        # Get starfruit orders
        starfruit_orders = self.compute_starfruit_orders(state)
        result["STARFRUIT"] = self.adjust_positions(starfruit_orders, "STARFRUIT")

        result["ORCHIDS"], conversions = self.computer_orchid_orders(state)

        # serialise data
        trader_data = self.serialize_to_string()

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data