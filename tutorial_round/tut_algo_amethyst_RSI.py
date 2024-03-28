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
    
    # This starfruit_cache stores the last 'starfruit_cache_num' of starfruit midprices
    starfruit_cache = []
    starfruit_time_cache = []
    starfruit_cache_num = 23 # change this value to adjust the 'lag'

    # Amethyst cache to store amethysts
    amethyst_cache = []
    amethyst_gain = 0
    amethyst_loss = 0
    amethyst_time_cache = []
    amethyst_cache_num = 14 # RSI generally operates on a 14* trade period 

    # Helper function to cache the midprice of a product
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

    # Gets the gains and losses for a amethyst based on RSI algorithm
    def get_amethyst_gains_losses(self, state: TradingState):
        '''
            GAIN: if current mid price > previous mid price = store currentprice - previous mid price
            LOSS: if current mid price < previous mid price = store previous mid price - currentprice
        '''
        order_depth: OrderDepth = state.order_depths["AMETHYSTS"]

        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]

        mid_price = (best_ask + best_bid)/2

        last_mid_price = self.amethyst_cache[len(self.amethyst_cache) - 2] # get the most previous midprice

        logger.print("Amethyst mid price: ", mid_price)
        logger.print("Amethyst last mid price: ", last_mid_price)

        self.amethyst_gain = 0
        self.amethyst_loss = 0

        for i in range(len(self.amethyst_cache) - 1):
            mid_price = self.amethyst_cache[i + 1]
            last_mid_price = self.amethyst_cache[i]
            if (mid_price > last_mid_price): 
                self.amethyst_gain += mid_price - last_mid_price
            else:
                self.amethyst_loss += last_mid_price - mid_price

        # if (mid_price > last_mid_price): 
        #     self.amethyst_gain += mid_price - last_mid_price
        # else:
        #     self.amethyst_loss += last_mid_price - mid_price
    
    # Assumes that length is not 0
    def get_amethyst_RSI(self):
        amethyst_prices_stored = len(self.amethyst_cache)

        amethyst_avg_gain = self.amethyst_gain / amethyst_prices_stored
        amethyst_avg_loss = self.amethyst_loss / amethyst_prices_stored
        logger.print("AMETHYST gain: ", amethyst_avg_gain, " ", amethyst_prices_stored)
        logger.print("AMETHYST loss: ", amethyst_avg_loss, " ", amethyst_prices_stored)

        if (amethyst_avg_loss == 0):
            return 50 # default RSI if there is no loss -- should this be 50?!?
        
        RS = amethyst_avg_gain / amethyst_avg_loss

        RSI = 100 - (100 / (1 + RS))

        logger.print("AMETHYST RSI: ", RSI)
        return RSI

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

    # This method is called at every timestamp -> it handles all the buy and sell orders, and outputs a list of orders to be sent
    def run(self, state: TradingState):

        # Dictionary that will end up storing all the orders of each product
        result = {}

        # Remove starfruit prices from cache if there are too many ()
        if (len(self.starfruit_cache) == self.starfruit_cache_num): 
            self.starfruit_cache.pop(0)
            self.starfruit_time_cache.pop(0)
        
        self.cache_product("STARFRUIT", state)

        # Do the same thing for amethyst
        if (len(self.amethyst_cache) == self.amethyst_cache_num): 
            self.amethyst_cache.pop(0)
            self.amethyst_time_cache.pop(0)
        
        self.cache_product("AMETHYSTS", state)

        # Do the actual buying and selling
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

            # Calculate RSI for amethysts
            if product == "AMETHYSTS": 
                #acceptable_price = 10000
                if (state.timestamp <= 100):
                    amethyst_RSI = 50
                else:
                    logger.print("Ameth GAIN: ", self.amethyst_gain)
                    logger.print("Ameth LOSS: ", self.amethyst_loss)
                    self.get_amethyst_gains_losses(state)
                    amethyst_RSI = self.get_amethyst_RSI()

            # Calculate prices for starfruit
            elif product == "STARFRUIT":
                # Price is based on regression prediction of next timestamp, otherwise average sum
                predicted_price = self.calculate_regression(self.starfruit_time_cache, self.starfruit_cache, state.timestamp + 100)

                # When the timestamp is not 0 and the price can be predicted with regression
                if (state.timestamp != 0 and predicted_price != -1): 
                    acceptable_price = round(predicted_price, 5)

                elif (state.timestamp == 0): # When the timestamp is 0, set price to 5000
                    acceptable_price = 5000

                else: # when the  price cannot be predicted with regression, then use moving average midprice
                    acceptable_price = round(sum(self.starfruit_cache)/self.starfruit_cache_num, 5)
               
                logger.print("Starfruit cache num", self.starfruit_cache_num)
                logger.print("Starfruit acceptable price ", acceptable_price)
                logger.print("Best ask: ", best_ask)
                logger.print("Best bid: ", best_bid)

            if product == "AMETHYSTS":
                # Do the BUYING 
                if len(order_depth.sell_orders) != 0:
                    if amethyst_RSI < 30:
                        logger.print(product, " BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
        
                # Do the SELLING
                if len(order_depth.buy_orders) != 0:
                    if amethyst_RSI > 70:
                        logger.print(product, " SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

            elif product == "STARFRUIT":
                # Do the BUYING 
                if len(order_depth.sell_orders) != 0:
                    if int(best_ask) < acceptable_price:
                        logger.print(product, " BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
        
                # Do the SELLING
                if len(order_depth.buy_orders) != 0:
                    if int(best_bid) > acceptable_price:
                        logger.print(product, " SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))

            # Add the orders of the corresponding product to result
            result[product] = orders
        
        trader_data = "" # String value holding Trader state data required. Delivered as TradingState.traderData on next execution.

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data