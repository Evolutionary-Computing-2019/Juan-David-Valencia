import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

start = datetime(2019, 1, 1)
end = datetime.now()

symbols_list = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
with open('sp500list', 'r') as fi:
    symbols_list = fi.read().split()[:40]

data = {}
for symbol in symbols_list:
    data[symbol] = web.DataReader(symbol, 'yahoo', start, end)['Close']

data = pd.DataFrame(data)

data.to_csv('data.csv')