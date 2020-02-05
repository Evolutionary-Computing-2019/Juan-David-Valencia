import numpy as np
from individual import Individual
from selection import tournament
from operators import FloatMutation, Crossover
from function import Markowitz, Sharpe
from functools import reduce
from operator import add
from scipy.linalg import expm


"""
Genetic implementation for Quantopian
"""

class HAEA:
    def __init__(self, lambda_, gen_size, data):
        self.lambda_ = lambda_
        self.gen_size = gen_size
        self.operators = [FloatMutation, Crossover]
        self.f = Sharpe(data, 1.0)

        self.state = []

    def initPopulation(self):
        population = []
        for _ in range(self.lambda_):
            genome = np.random.uniform(0.0, 100.0, self.gen_size)
            genome = genome/sum(genome)

            rates = np.ones(len(self.operators))
            rates = rates / sum(rates) # Uniform probability
            fitness = self.f.calculate(genome)
            individual = Individual(genome=genome, fitness=fitness, rates=rates)
            population.append(individual)
        return population

    
    def terminationCondition(self, t, P):
        return t > 200

    def op_select(self, rates):
        return np.random.choice(range(len(self.operators)), p=rates)

    def parent_selection(self, P, ind, oper):
        n = self.operators[oper].arity
        parents = [ind]
        if n > 1:
            parents.extend(np.random.choice(P, n - 1))
        return parents

    def apply(self, oper, parents):
        return self.operators[oper].eval(*parents, 1)

    def best(self, offspring, ind):
        return max( offspring + [ind] , key=lambda i: i.fitness )

    def eval(self):
        t = 0
        P = self.initPopulation(  )
        while not self.terminationCondition(t, P):
            P_next = []
            for ind in P:
                rates = ind.rates #self.extract_rates(ind)
                delta = np.random.normal(0, 0.01)
                oper = self.op_select(rates)
                parents = self.parent_selection(P, ind, oper)
                offspring = self.apply(oper, parents)

                # Calculate the fitness
                for off in offspring:
                    off.fitness = self.f.calculate(off.genome)

                child = self.best(offspring, ind) #???
                if child.fitness > ind.fitness:
                    rates[oper] = (1.0 + delta)*rates[oper]
                else:
                    rates[oper] = (1.0 - delta)*rates[oper]

                # Normalize
                rates = np.array(rates)/sum(rates) # self.normalize_rates(rates)
                
                # Set rates
                child.rates = rates # self.set_rates( child, rates )
                
                P_next.append( child )
            P = P_next
            
            self.state.append( max(P, key=lambda ind: ind.fitness).fitness )
            t = t+1


class SimulatedAnnealing:
    def __init__(self, lambda_, sa, gen_size, data):
        self.lambda_ = lambda_
        self.gen_size = gen_size
        self.f = Sharpe(data, 1.0)
        self.state = []
        self.sa = sa

    def initPopulation(self):
        population = []
        for _ in range(self.lambda_):
            genome = np.random.uniform(0.0, 100.0, self.gen_size)
            genome = genome/sum(genome)

            fitness = self.f.calculate(genome)
            individual = Individual(genome=genome, fitness=fitness)
            population.append(individual)

        return population

    def terminationCondition(self, t, P):
        return t > 200

    def replace(self, ind, new_ind, t):
        k = 0.001*(10*t + 1)
        temperature = np.abs(np.sin(k)/k)
        if self.sa:
            if new_ind.fitness >= ind.fitness or np.random.choice(range(2), p=[1-temperature, temperature]):
                return new_ind
        else:
            if new_ind.fitness >= ind.fitness:
                return new_ind
        return ind

    def create_offspring(self, ind):
        delta = 0.001 * np.random.normal(0, 2, self.gen_size)
        new_genome = ind.genome + delta
        new_genome = np.abs(new_genome)/sum(new_genome)
        new_ind = Individual(genome=new_genome, fitness=np.nan)
        new_ind.fitness = self.f.calculate(new_ind.genome)
        return new_ind

    def eval(self):
        P = self.initPopulation()
        t = 0
        while not self.terminationCondition(t, P):
            P_next = []
            for ind in P:
                son = self.create_offspring(ind)
                new_ind = self.replace(ind, son, t)
                P_next.append(new_ind)
                P_next.append(son)
            P = sorted(P_next, key=lambda ind: -ind.fitness)[:self.lambda_]
            t = t+1
            self.state.append( max(P, key=lambda ind: ind.fitness).fitness )

class HillClimbing(SimulatedAnnealing):
    def __init__(self, lambda_, gen_size, data):
        super().__init__(lambda_, False, gen_size, data)


class SelfAdaptation:
    def __init__(self, lambda_, gen_size, data):
        self.n = np.random.randint(100)
        self.lambda_ = lambda_
        self.mu = int(np.ceil(lambda_ / 4))
        self.tau = 1/np.sqrt( self.n )
        self.tau_i = 1/(np.sqrt(self.n)**0.25)
        self.f = Sharpe(data, 1.0)
        self.gen_size = gen_size

        self.state = []

    def terminationCondition(self, t):
        return t > 200

    def sel_mu_best(self, P):
        P.sort(key=lambda k: k.fitness, reverse=True)
        return P[:self.mu]

    def eval(self):
        vec_x = np.random.normal(0, 1, self.gen_size)
        vec_x = vec_x / sum(vec_x)
        vec_sigma = np.random.normal(0, 1, self.gen_size)
        t = 0
        while not self.terminationCondition(t):
            lambda_elements = []
            for _ in range(self.lambda_):
                xi_k = self.tau * np.random.normal(0, 1)
                vec_xi_k = self.tau_i * np.random.normal(0, 1, size=self.gen_size)
                vec_z_k = np.random.normal(0, 1, size=self.gen_size)

                vec_sigma_k = np.multiply( vec_sigma, np.exp(vec_xi_k) ) * np.exp(xi_k) 
                vec_x_k = vec_x + np.multiply(vec_sigma_k , vec_z_k)
                vec_x_k = vec_x_k / sum(vec_x_k)
                lambda_elements.append( Individual( genome=vec_x_k, sigma=vec_sigma_k, fitness=self.f.calculate(vec_x_k) ) )

            P = self.sel_mu_best(lambda_elements)

            vec_sigma = (1./self.mu)* sum(map(lambda ind: ind.sigma, P))
            vec_x = (1./self.mu) * sum(map(lambda ind: ind.genome, P))
            vec_x = vec_x / sum(vec_x)
            t += 1
            self.state.append( max(P, key=lambda ind: ind.fitness).fitness )


class Derandomize:
    def __init__(self, lambda_, gen_size, data):
        self.lambda_ = lambda_
        self.gen_size = gen_size

        self.tau = 1/3
        self.d = np.sqrt(gen_size)
        self.d_i = gen_size
        self.f = Sharpe(data, 1.0)
        self.state = []

    def terminationCondition(self, t):
        return t > 200

    def selectBest(self, P):
        return max(P, key=lambda k: k.fitness )

    def eval(self):
        t = 0
        vec_x = np.random.normal(0, 1, self.gen_size)
        vec_sigma = np.random.random(self.gen_size)  
        P = []
        while not self.terminationCondition(t):
            for _ in range(self.lambda_):
                xi_k = self.tau * np.random.normal(0,1)
                vec_z_k = np.random.normal(0, 1, self.gen_size)
                vec_x_k = np.multiply(vec_x + np.exp(xi_k), np.multiply( vec_sigma, vec_z_k ) )
                vec_sigma_k = np.multiply(vec_sigma,  np.exp( (1/self.d_i) * (abs(vec_z_k)/np.average(abs(np.random.normal(0, 1, 100))) - np.ones(self.gen_size)) ) ) * np.exp((1/self.d) * xi_k)
                P.append(Individual(genome=vec_x_k, sigma=vec_sigma_k, fitness=self.f.calculate(vec_x_k)))

            best = self.selectBest(P)
            self.state.append(best.fitness)
            vec_x = best.genome
            vec_sigma = best.sigma
            t += 1