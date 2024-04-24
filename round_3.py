import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import numpy as np
import pandas as pd
import math

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

    POSITION_LIMIT = {'STARFRUIT' : 20, 
                      'AMETHYSTS' : 20, 
                      'ORCHIDS': 100, 
                      'CHOCOLATE': 250, 
                      'STRAWBERRIES': 350, 
                      'ROSES': 60, 
                      "GIFT_BASKET": 60,
                      "COCONUT": 300,
                      "COCONUT_COUPON": 600}

    # Stores how many values to cache for each product 
    product_cache_num = {"STARFRUIT" : 20}

    # Stores last starfruit midprices - used for regression
    starfruit_cache = []

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

    # Handles caching of a product by removing items from cache before overflow, and appending new midprices to the cache
    def handle_starfruit_cache(self, product: Symbol, state: TradingState, cache: list[int]):
        cache_num = self.product_cache_num[product]
        
        # Remove prices from cache if there are too many
        if (len(cache) == cache_num): 
            cache.pop(0)

        order_depth: OrderDepth = state.order_depths[product]

        # Extract the best ask and bid
        best_ask, _ = list(order_depth.sell_orders.items())[0]
        best_bid, _ = list(order_depth.buy_orders.items())[0]

        # Add the product midprice and timestamp to relevant cache
        cache.append((best_ask + best_bid)/2)

    # Calculates regression when given the times and prices, and a timestamp to predict the price of
    def calculate_regression(self, prices: list[int], timestamp: int):
        times = [x for x in range(len(prices))]
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
    def get_price_regression(self, cache: list[int], forecast: int):
        # Price is based on regression prediction of next timestamp, otherwise average sum
        predicted_price = self.calculate_regression(cache, len(cache) * forecast)

        # When the timestamp is not 0 and the price can be predicted with regression
        if (predicted_price != -1): 
            return predicted_price
        else: # when the price cannot be predicted with regression, then use average midprice
            return sum(cache)/len(cache)

    def get_spread(self, cache):
        y = 0.1
        k = 1.5
        sd = np.std(cache)

        return y*(sd**2) + 2/y * math.log(1 + y/k)
    
    # Returns weighted midprice from order book
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
        acceptable_price_regres = self.get_price_regression(self.starfruit_cache, forecast = 1)
        
         # Calculate price of last 5 starfruit
        acceptable_price_avg = sum(self.starfruit_cache[-5:])/len(self.starfruit_cache[-5:])

        logger.print(f"{product} acceptable regres price: {acceptable_price_regres}\navg price: {acceptable_price_avg}")
        logger.print(f"{product} best ask: {best_ask}")
        logger.print(f"{product} best bid: {best_bid}")

        ask_amount = min(-best_ask_amount, self.POSITION_LIMIT[product] - state.position.get(product, 0))
        bid_amount = max(-best_bid_amount, -self.POSITION_LIMIT[product] - state.position.get(product, 0))

        # Do the BUYING 
        if len(order_depth.sell_orders) != 0:
            # Buy based on average price
            if best_ask <= acceptable_price_avg: 
                logger.print(product, " BUY avg", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, ask_amount))

             # Buy based on regression price
            elif best_ask <= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]:
                logger.print(product, " BUY regres", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, ask_amount))

        # Do the SELLING
        if len(order_depth.buy_orders) != 0:
            # Sell based on average price
            if best_bid >= acceptable_price_avg:
                logger.print(product, " SELL avg", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, bid_amount))

            # Sell based on regression price
            elif best_bid >= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]:
                logger.print(product, " SELL regres", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, bid_amount))

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

        if not (best_ask <= acceptable_price_avg or best_ask <= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]):
            # Send a buy order (bots will sell to us at this price and we are looking to buy)
            orders.append(Order(product, min(int(price - spread), best_bid + undercut), bid_amount)) 

        if not (best_bid >= acceptable_price_avg or best_bid >= acceptable_price_regres and state.timestamp >= 100 * self.product_cache_num[product]):
            # Send a sell order (bots will buy from us at this price and we are looking to sell)
            orders.append(Order(product, max(int(price + spread), best_ask - undercut), -ask_amount)) 

        return orders

    def compute_orchid_orders(self, state: TradingState):
        product = "ORCHIDS"
        orders: List[Order] = []

        conversions = 0
        bid_amount = 100
        ask_amount = 100
        observation = state.observations.conversionObservations['ORCHIDS']

        # buying
        spread = 3
        offer_price = state.observations.conversionObservations['ORCHIDS'].bidPrice - observation.exportTariff - spread
        orders.append(Order(product, int(offer_price), bid_amount))

        # selling
        
        spread = 2 #min(abs(observation.importTariff) + 1, 2) # used 2.5 in round 2 -> This should be calculated based on something BUG OPTIMISE
        offer_price = state.observations.conversionObservations['ORCHIDS'].askPrice + observation.importTariff + spread 
        orders.append(Order(product, math.ceil(offer_price), -ask_amount))

        conversions = -state.position.get(product, 0)

        return orders, conversions

    def roses_bot_signal(self, state: TradingState):
        product = "ROSES"
        order_depth: List[Trade] = state.market_trades.get(product, []) 
        if order_depth != []:
            for trade in order_depth:
                if trade.seller == "Rhianna":
                    return "sell"
                if trade.buyer == "Rhianna":
                    return "buy"
        return None

    def choc_bot_signal(self, state: TradingState):
        product = "CHOCOLATE"
        order_depth: List[Trade] = state.market_trades.get(product, []) 
        if order_depth != []:
            for trade in order_depth:
                if trade.seller == "Vladimir":     
                    return "sell"
                if trade.buyer == "Vladimir":
                    return "buy"
        return None
    
    def straw_bot_signal(self, state: TradingState):
        product = "STRAWBERRIES"
        order_depth: List[Trade] = state.market_trades.get(product, []) 
        if order_depth != []:
            for trade in order_depth:
                if trade.seller == "Vlad":
                    return "sell"
                if trade.buyer == "Vlad":
                    return "buy"
        return None


    def compute_basket_orders(self, state: TradingState):
        products = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        orders = {"CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}
        sell_orders = {}
        buy_orders = {}

        # gift basket = 6 straw, 4 chocs, 1 rose
        number_in_basket = {"CHOCOLATE": 4, "STRAWBERRIES": 6, "ROSES": 1}
        expected_basket_mid_price = 0

        for product in products:

            order_depth: OrderDepth = state.order_depths[product]

            market_sell_orders = list(order_depth.sell_orders.items())
            market_buy_orders = list(order_depth.buy_orders.items())

            sell_orders[product] = market_sell_orders
            buy_orders[product] = market_buy_orders

            best_ask, best_ask_amount = market_sell_orders[0]
            best_bid, best_bid_amount = market_buy_orders[0]

            mid_price = (best_ask + best_bid)/2

            if product != "GIFT_BASKET":
                expected_basket_mid_price += number_in_basket[product] * mid_price

            else: # For GIFT_BASKETS:
                  # Market take
                logger.print(f"Expected {product} price = {expected_basket_mid_price}\nReal {product} price = {mid_price}\nBest ask: {best_ask}\nBest bid: {best_bid}")

                expected_premium = 380
                difference = mid_price - expected_basket_mid_price
                spread = 35

                logger.print(f"expec prem {expected_premium}")
                logger.print(f"expec basket price {expected_basket_mid_price}")
                logger.print("spread is", spread)

                ask_amount = min(-best_ask_amount, self.POSITION_LIMIT[product] - state.position.get(product, 0))
                bid_amount = max(-best_bid_amount, -self.POSITION_LIMIT[product] - state.position.get(product, 0))

                # Send a buy order (bots will sell to us at this price and we are looking to buy)
                if difference < expected_premium - spread: # best results with 40 and 380 as premium spread
                    logger.print(f"market buy {best_ask}")
                    orders[product].append(Order(product, best_ask, ask_amount))
                else:
                    orders[product].append(Order(product, int(expected_basket_mid_price + expected_premium - 20), self.POSITION_LIMIT[product] - state.position.get(product, 0)))

                # Send a sell order (bots will buy from us at this price and we are looking to sell)
                if difference > expected_premium + spread:
                    logger.print(f"market sell {best_bid}")
                    orders[product].append(Order(product, best_bid, bid_amount))
                else:
                    orders[product].append(Order(product, math.ceil(expected_basket_mid_price + expected_premium + 20), -self.POSITION_LIMIT[product] - state.position.get(product, 0)))

                for item in products:
                    best_bid, best_bid_amount = buy_orders[item][0]
                    best_ask, best_ask_amount = sell_orders[item][0]
                    ask_amount = min(-best_ask_amount, self.POSITION_LIMIT[item] - state.position.get(item, 0))
                    bid_amount = max(-best_bid_amount, -self.POSITION_LIMIT[item] - state.position.get(item, 0))

                    prod_mid = (best_ask + best_bid)/2

                    if item == "CHOCOLATE":
                        if prod_mid > mid_price - 62728.5435 + 1000: # and not (difference < expected_premium - spread):
                        # if self.choc_bot_signal(state) == "sell":
                            orders[item].append(Order(item, best_ask, ask_amount)) # buy
                        if prod_mid < mid_price - 62728.5435 - 500: # and not (difference > expected_premium + spread):
                        # if self.choc_bot_signal(state) == "buy":
                            orders[item].append(Order(item, best_bid, bid_amount)) # sell

                    if item == "ROSES":
                        if self.roses_bot_signal(state) == "buy":
                             orders[item].append(Order(item, best_ask, ask_amount)) 
                        if self.roses_bot_signal(state) == "sell":
                            orders[item].append(Order(item, best_bid, bid_amount))

                    if item == "STRAWBERRIES":
                        if self.straw_bot_signal(state) == "buy":
                             orders[item].append(Order(item, best_ask, ask_amount)) 
                        if self.straw_bot_signal(state) == "sell":
                            orders[item].append(Order(item, best_bid, bid_amount))

        return orders["CHOCOLATE"], orders["STRAWBERRIES"], orders["ROSES"], orders["GIFT_BASKET"]

    def norm_cdf(self, x):
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))

    def black_scholes(self, mid_price: float):
        S = mid_price # Spot price
        K = 10000  # Strike price
        T = 246/252  # Time to expiration (in years)
        r = 0.001943 # Risk-free interest rate
        sigma = 0.161945 #0.16195 #if (len(self.coconut_diffs) < 5) else np.std(self.coconut_diffs) #0.161 #0.16195  #0.161940117 # Volatility
        logger.print(sigma)

        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Call option
        c_price = S * np.exp(-r * T) * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
        
        # Put option
        # p_price = K * np.exp(-r * T) * self.norm_cdf(-d2) - S * np.exp(-r * T) * self.norm_cdf(-d1)

        return c_price #, p_price

    def compute_coupon_orders(self, state):
        products = ["COCONUT", "COCONUT_COUPON"]
        sell_orders = {"COCONUT": [], "COCONUT_COUPON": []}
        buy_orders = {"COCONUT": [], "COCONUT_COUPON": []}
        mid_prices = {"COCONUT": 0, "COCONUT_COUPON": 0}
        orders = {"COCONUT": [], "COCONUT_COUPON": []}
        
        for product in products:
            order_depth: OrderDepth = state.order_depths[product]

            market_sell_orders = list(order_depth.sell_orders.items())
            market_buy_orders = list(order_depth.buy_orders.items())

            if len(order_depth.buy_orders) != 0 and len(order_depth.sell_orders) != 0:
                market_buy_orders = list(order_depth.buy_orders.items())
                market_sell_orders = list(order_depth.sell_orders.items())

                buy_orders[product] = market_buy_orders
                sell_orders[product] = market_sell_orders

                best_bid, best_bid_amount = market_buy_orders[0]
                best_ask, best_ask_amount = market_sell_orders[0]

                mid_prices[product] = (best_ask + best_bid)/2

        product = "COCONUT"
        if len(buy_orders[product]) == 0 or len(sell_orders[product]) == 0:
            coc_mid_price = 10000
        else:
            coc_mid_price = mid_prices[product]

        c_price = self.black_scholes(coc_mid_price)

        logger.print(c_price)

        product = "COCONUT"
        if len(sell_orders[product]) != 0 and len(buy_orders[product]) != 0:
            coc_best_ask, best_ask_amount = sell_orders[product][0]
            coc_best_bid, best_bid_amount = buy_orders[product][0]

            coc_ask_amount = min(-best_ask_amount, self.POSITION_LIMIT[product]//2 - state.position.get(product, 0))
            coc_bid_amount = max(-best_bid_amount, -self.POSITION_LIMIT[product] - state.position.get(product, 0))

        product = "COCONUT_COUPON"
        if len(sell_orders[product]) != 0 and len(buy_orders[product]) != 0:
            best_ask, best_ask_amount = sell_orders[product][0]
            best_bid, best_bid_amount = buy_orders[product][0]

        spread = 5
        # buying
        if len(sell_orders[product]) != 0:
            amount = min(-best_ask_amount, self.POSITION_LIMIT[product] - state.position.get(product, 0))
            if best_ask <= c_price: # take
                orders[product].append(Order(product, best_ask, amount))
                logger.print(product, " BUY", str(amount) + "x", best_ask)
                
                # sell hedge coconuts
                if state.position.get(product, 0) > 100:
                    orders["COCONUT"].append(Order("COCONUT", coc_best_bid, coc_bid_amount))

            else: # market make
                orders[product].append(Order(product, round(c_price) - spread, abs(self.POSITION_LIMIT[product] - state.position.get(product, 0))))

        # Do the SELLING
        if len(buy_orders[product]) != 0:
            amount = max(-best_bid_amount, -self.POSITION_LIMIT[product] - state.position.get(product, 0))
            if best_bid >= c_price: # take
                orders[product].append(Order(product, best_bid, amount))
                logger.print(product, " SELL", str(-amount) + "x", best_bid)

                # take buy hedge coconuts
                if state.position.get(product, 0) < -100:
                    orders["COCONUT"].append(Order("COCONUT", coc_best_ask, coc_ask_amount))

            else: # market make
                orders[product].append(Order(product, round(c_price) + spread, -self.POSITION_LIMIT[product] - state.position.get(product, 0)))

        return orders["COCONUT"], orders["COCONUT_COUPON"]

    def run(self, state: TradingState):
        # update cache only if information is lost
        if state.traderData != "" and self.starfruit_cache == []:
            self.deserialize_data(state.traderData)
        
        # Dictionary that will end up storing all the orders of each product
        result = {}

        # self.handle_starfruit_cache('STARFRUIT', state, self.starfruit_cache)

        # amethyst_orders = self.compute_amethyst_orders(state)
        # result["AMETHYSTS"] = amethyst_orders

        # starfruit_orders = self.compute_starfruit_orders(state)
        # result["STARFRUIT"] = starfruit_orders

        # result["ORCHIDS"], conversions = self.compute_orchid_orders(state)

        result["CHOCOLATE"], result["STRAWBERRIES"], result["ROSES"] , _ = self.compute_basket_orders(state) # this fucking sucks why

        # result["COCONUT"], result["COCONUT_COUPON"] = self.compute_coupon_orders(state)
        
        # serialise data
        trader_data = self.serialize_to_string()
        conversions = 0
        logger.flush(state, result, 0, trader_data)
        return result, conversions, trader_data
    