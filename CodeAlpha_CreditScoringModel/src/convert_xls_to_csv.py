import pandas as pd

train = pd.read_excel(
    "data/train-FIN_ANA_DATA .xls"
)
train.to_csv(
    "data/train.csv",
    index=False
)

test = pd.read_excel(
    "data/test-FIN_ANA_DATA .xls"
)
test.to_csv(
    "data/test.csv",
    index = False
)