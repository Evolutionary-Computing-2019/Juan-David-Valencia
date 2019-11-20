import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms import HAEA, SimulatedAnnealing, HillClimbing


start = datetime(2015, 1, 1)
end = datetime.now()

data = pd.read_csv('data.csv', index_col='Date')


h = HAEA(100, data.shape[1], data)
h.eval()
plt.subplot(131)
plt.plot(h.state[:200])



sa = SimulatedAnnealing(100, True, data.shape[1], data)
sa.eval()
plt.subplot(132)
plt.plot(sa.state[:200])


hc = HillClimbing(100, data.shape[1], data)
hc.eval()
plt.subplot(133)
plt.plot(hc.state[:200])


plt.show()