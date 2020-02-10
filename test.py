import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms import HAEA, SimulatedAnnealing, HillClimbing, SelfAdaptation, Derandomize, NSGAII


start = datetime(2015, 1, 1)
end = datetime.now()

data = pd.read_csv('data.csv', index_col='Date')


h = HAEA(100, data.shape[1], data)
h.eval()
plt.subplot(211)
plt.plot(h.state[:200])


"""
sa = SimulatedAnnealing(100, True, data.shape[1], data)
sa.eval()
plt.subplot(222)
plt.plot(sa.state[:200])


hc = HillClimbing(100, data.shape[1], data)
hc.eval()
plt.subplot(223)
plt.plot(hc.state[:200])


sa = SelfAdaptation(100, data.shape[1], data)
sa.eval()
plt.subplot(224)
plt.plot(sa.state[:200])


sa = Derandomize(100, data.shape[1], data)
sa.eval()
plt.subplot(224)
plt.plot(sa.state[:200])

"""
ngsa2 = NSGAII(100, data.shape[1], data)
ngsa2.eval()
plt.subplot(212)
plt.plot(ngsa2.state[:200])


plt.show()

