import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_name):
    info = open(file_name, 'r').read()
    info = list(map(lambda x: list(map(float, x.split())), info.split('\n')[:-1]))
    return pd.DataFrame(info)


list_data = []

algorithms = ['haea', 'hill', 'sa', 'selfadaptation']
titles = {
        'haea': 'HAEA',
        'hill': 'Hill Climbing',
        'sa': 'Simulated Annealing',
        'selfadaptation': 'Self Adaptation'
        }
for alg in algorithms:
    list_data.append((alg, read_data(alg)))

for name, data in list_data:
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(data.max(), label='Max')
    ax.plot(data.min(), label="Min")
    ax.plot(data.mean(), label="Mean")
    ax.plot(data.median(), label="Median")
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(titles[name])
    ax.legend()
    plt.show()
