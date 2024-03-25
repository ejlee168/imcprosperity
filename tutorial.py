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
    starfruit_cache_num = 15 # change this value to adjust the 'lag'

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

        # Do the actual buying and selling
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Sets acceptable prices to buy
            if product == "AMETHYSTS": 
                acceptable_price = 10000 # Amethyst price is static, therefore static acceptable price
            elif product == "STARFRUIT":
                # Price is based on regression prediction of next timestamp, otherwise average sum
                predicted_price = self.calculate_regression(self.starfruit_time_cache, self.starfruit_cache, state.timestamp + 100)

                if (state.timestamp != 0 and predicted_price != -1):
                    acceptable_price = predicted_price

                elif (state.timestamp == 0):
                    acceptable_price = 5000
                    
                else:
                    acceptable_price = sum(self.starfruit_cache)/self.starfruit_cache_num

                logger.print("Starfruit acceptable price ", acceptable_price)

            # Do the BUYING 
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    logger.print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            # Do the SELLING
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            # Add the orders of the corresponding product to result
            result[product] = orders
        
        trader_data = "" # String value holding Trader state data required. Delivered as TradingState.traderData on next execution.

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data