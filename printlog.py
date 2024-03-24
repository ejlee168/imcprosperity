import pandas as pd
import numpy as np


headers = [
    "Day", "Timestamp", "Product", "Bid Price 1", "Bid Volume 1",
    "Bid Price 2", "Bid Volume 2", "Bid Price 3", "Bid Volume 3",
    "Ask Price 1", "Ask Volume 1", "Ask Price 2", "Ask Volume 2",
    "Ask Price 3", "Ask Volume 3", "Mid Price", "Profit & Loss"
]

head = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
head = head.split(";")
new_csv = pd.DataFrame(columns=head)

with open("log1.txt", "r") as log:
    for line in log:
        line = line.strip()
        line = line.split(";")
        print(line)
        new_csv.loc[len(new_csv)] = line

new_csv.to_excel("log.xlsx")
