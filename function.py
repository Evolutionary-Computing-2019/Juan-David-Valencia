# IN THIS FILE WE DEFINE THE FUNCTION TO OPTIMIZE
# IDEA in mutation normalize the genome
import numpy as np
from abc import ABC, abstractmethod


class Function(ABC):
    @abstractmethod
    def calculate(self):
        pass


class Markowitz(Function):
    def __init__(self, assets_data, max_weight):
        self.data = assets_data
        self.max_weight = max_weight

    def calculate(self, representation):
        fitness = 0.0
        if sum(representation) == 1.0:
            # NOTE 
            out_constraint = len(list(filter(lambda w: w > self.max_weight, representation))) > 0
            if out_constraint:
                fitness = 0.0
            else:
                # Calculate portfolio
                returns = np.asmatrix(np.diff(self.data.as_matrix(), axis=0)/self.data.as_matrix()[:-1])
                returns = returns.T

                weights = np.asmatrix(representation).T

                P = np.asmatrix(np.mean(returns, axis=1))
                C = np.asmatrix(np.cov(returns))

                mu = float(P.T*weights)
                sigma = float(np.sqrt(weights.T * C * weights))

                fitness =  mu
        return fitness

class Sharpe(Function):
    def __init__(self, assets_data, max_weight):
       self.data = assets_data
       self.max_weight = max_weight

    def calculate(self, representation):
        weights = np.asmatrix(representation).T

        returns = np.asmatrix(np.diff(self.data.as_matrix(), axis=0)/self.data.as_matrix()[:-1])
        returns = returns.T

        P = np.asmatrix(np.mean(returns, axis=1))
        ER = np.dot(P.T, weights)
        sharpe = ER[0, 0]/np.mean(np.std(P))
        # Constraints
        contr = (sum(representation) - 1)**2

        for g in representation:
            contr += max(0, g - 1)**2
            contr += max(0, -g)**2

        return sharpe - 100*contr
     