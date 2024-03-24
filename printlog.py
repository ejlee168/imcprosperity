import pandas as pd
import numpy as np

def replace(line: str):
    return line.replace(";", ",")

headers = [
    "Day", "Timestamp", "Product", "Bid Price 1", "Bid Volume 1",
    "Bid Price 2", "Bid Volume 2", "Bid Price 3", "Bid Volume 3",
    "Ask Price 1", "Ask Volume 1", "Ask Price 2", "Ask Volume 2",
    "Ask Price 3", "Ask Volume 3", "Mid Price", "Profit & Loss"
]

new_csv = pd.DataFrame(columns=headers)

with open("log1.txt", "r") as log:
    for line in log:
        line = replace(line)
        new_csv.loc[len(new_csv)] = line

new_csv.to_excel("log.xlsx")
