import numpy as np
from individual import Individual
from abc import ABC, abstractmethod


class Operator:
    @abstractmethod
    def eval(self):
        pass


class FloatMutation(Operator):
    arity = 1

    @classmethod
    def eval(cls, ind, p):
        if 'rates' in ind.__dict__:
            c_ind = Individual(genome=np.copy(ind.genome), fitness=np.nan, rates=np.copy(ind.rates))
        else:
            c_ind = Individual(genome=np.copy(ind.genome), fitness=np.nan)
        if np.random.uniform() < p:
            for n in range(len(c_ind.genome)):
                c_ind.genome[n] += np.random.normal(0, 0.01)
                c_ind.genome[n] = max(0.0, c_ind.genome[n])

        c_ind.genome = c_ind.genome/sum(c_ind.genome)
        return [c_ind]

class Crossover(Operator):
    arity = 2

    @classmethod
    def eval(cls, ind, other, p):

        if not 'rates' in ind.__dict__:
            ind.rates = np.ones(1)

        genome_1, genome_2 = [], []
        if np.random.uniform() < p:
            dice = np.random.randint(1, len(ind.genome))
            genome_1 = np.concatenate([ind.genome[:dice], other.genome[dice:]])
            genome_2 = np.concatenate([other.genome[:dice], ind.genome[dice:]])
        else:
            genome_1 = ind.genome
            genome_2 = other.genome
        #NOTE New rates for crossover
        genome_1 = np.array(genome_1)/sum(genome_1)
        genome_2 = np.array(genome_2)/sum(genome_2)

        rates = np.ones(len(ind.rates))
        rates = rates/sum(rates)

        return [Individual(genome=genome_1, fitness=np.nan, rates=rates), \
                Individual(genome=genome_2, fitness=np.nan, rates=rates)]


