import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

start = datetime(2015, 1, 1)
end = datetime.now()

symbols_list = ['ORCL', 'TSLA', 'IBM','YELP', 'MSFT']

data = {}
for symbol in symbols_list:
    data[symbol] = web.DataReader(symbol, 'yahoo', start, end)['Close']

data = pd.DataFrame(data)

data.to_csv('data.csv')