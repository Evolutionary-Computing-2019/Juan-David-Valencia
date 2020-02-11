import sys
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms import HAEA, SimulatedAnnealing, HillClimbing, SelfAdaptation, Derandomize, NSGAII


start = datetime(2015, 1, 1)
end = datetime.now()

data = pd.read_csv('data.csv', index_col='Date')

if sys.argv[1] == 'haea':
    h = HAEA(66, data.shape[1], data)
    h.eval()
    print(h.f.count)
    #plt.subplot(231)
    plt.plot(h.state[:200])

elif sys.argv[1] == 'sa':
    sa = SimulatedAnnealing(100, True, data.shape[1], data)
    sa.eval()
    print(sa.f.count)
    #plt.subplot(232)
    plt.plot(sa.state[:200])

elif sys.argv[1] == 'hill':
    hc = HillClimbing(100, data.shape[1], data)
    hc.eval()
    print(hc.f.count)
    #plt.subplot(233)
    plt.plot(hc.state[:200])

elif sys.argv[1] == 'selfadaptation':
    sa = SelfAdaptation(100, data.shape[1], data)
    sa.eval()
    print(sa.f.count)
    #plt.subplot(234)
    plt.plot(sa.state[:200])

elif sys.argv[1] == 'derandomize':
    sa = Derandomize(100, data.shape[1], data)
    sa.eval()
    print(sa.f.count)
    #plt.subplot(235)
    plt.plot(sa.state[:200])

elif sys.argv[1] == 'nsga':
    ngsa2 = NSGAII(100, data.shape[1], data)
    ngsa2.eval()
    #plt.subplot(236)
    plt.plot(ngsa2.state[:200])

plt.show()

