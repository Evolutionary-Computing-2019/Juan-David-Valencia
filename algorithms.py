import random
import numpy as np
from individual import Individual
from selection import tournament
from operators import FloatMutation, Crossover
from function import Markowitz, Sharpe, SharpeV2
from functools import reduce
from operator import add
from collections import defaultdict
from sortNondominated import sortNondominated as fast_non_dominated_sort
from CrowdingDistance import CrowdingDist as crowfonding_distance_assignment


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



class NSGAII:
    def __init__(self, N, gen_size, data):
        self.N = N
        self.gen_size = gen_size
        self.f = SharpeV2(data, 1.0)
        self.f2 = Sharpe(data, 1.0)

        self.state = []

    def dominates(self, x_1, x_2):
        """ This function returns True, if x_1 dominates x_2. False otherwise.
        x_1: List of fitnesses
        x_2: List of fitnesses
        >>> dominates([0.5, 0.0], [1.0, 0.0])
        True
        >>> dominates([2.0, 3.0], [2.0, 2.0])
        False
        """
        # x_1 is worst than x_2
        for i, f in enumerate(x_1):
            if f > x_2[i]: return False
        for i, f in enumerate(x_1):
            if f < x_2[i]: return True
        return False

    def crowding_distance_assignment(self, F):
        """
        This function assigns a distance on the solutions of a specific 
        front.

        Recieves a list of individuals
        """
        length = len(F)
        F_dist = {}
        for i, ind in enumerate(F):
            F_dist[i] = 0
        for m, _ in enumerate(range(3)):
            F.sort(key=lambda ind: ind.fitness[m])
            F_dist[0] = F_dist[length - 1] = float('inf')
            for i in range(1, length - 1):
                F_dist[i] = F_dist[i] + (F[i + 1].fitness[m] - F[i - 1].fitness[m])
        Distances = {}
        for i, ind in enumerate(F):
            Distances[ind] = F_dist[i]
        return Distances

    def fast_nondominated_sort(self, P):
        """
        This algorithm finds the Paretto frontiers
        P is a list of individuals [Individual(), Individual()] with a 
        parameter call fitness with a list of values.
        """
        F = defaultdict(set) 
        S = defaultdict(set)
        n = defaultdict(int)
        for p in P:
            for q in P:
                if self.dominates(p.fitness, q.fitness):
                    S[p] = S[p].union({q})
                elif self.dominates(q.fitness, p.fitness):
                    n[p] += 1
            if n[p] == 0:
                F[0] = F[0].union({p})
        i = 0
        while F[i]:
            H = set()
            for p in F[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        H = H.union({q})
            i += 1
            F[i] = H
        F.pop(i)
        return F

    def terminationCondition(self, t):
        return t > 200

    def initialize_population(self):
        genomes = [ np.random.normal(0.0, 100.0, self.gen_size) for _ in range(self.N) ]
        for ind, g in enumerate(genomes):
            genomes[ind] = g/sum(g)
            genomes[ind] = np.max(np.vstack((genomes[ind], np.zeros(self.gen_size))), axis=0)
        P = []
        for g in genomes:
            f1, f2, f3 = self.f.calculate(g)
            P.append( Individual( genome=g, fitness=(f1, f2, f3) ) )
        return P

    def make_new_pop(self, P):
        # Mutations
        Q = []
        for ind in random.choices(P, k=len(P)//2):
            new_ind = FloatMutation.eval(ind, 1)[0]
            new_ind.fitness = self.f.calculate(new_ind.genome)  
            Q.append(new_ind)

        for ind in random.choices(P, k=len(P)//4):
            new_ind1, new_ind2 = Crossover.eval(ind, random.choice(P), 1)
            new_ind1.fitness = self.f.calculate(new_ind1.genome)
            new_ind2.fitness = self.f.calculate(new_ind2.genome)
            Q.extend([new_ind1, new_ind2])
        return Q

    def eval(self):
        t = 0
        P = self.initialize_population()
        Q = []
        
        while not self.terminationCondition(t):
            R = P + Q
            F = self.fast_nondominated_sort(R)
            P_next = []
            i = 0
            Distances = {}
            while len(P_next) < self.N:
                dist_current = self.crowding_distance_assignment(list(F[i]))
                for k in dist_current:
                    Distances[k] = dist_current[k]
                for individual in sorted(F[i], key=lambda ind: Distances[ind], reverse=True):
                    if len(P_next) < self.N:
                        P_next.append(individual)
                i += 1
            #P_next.sort(key=lambda ind: Distances[ind], reverse=True)
            Q_next = self.make_new_pop(P_next)
            t += 1
            P = P_next
            Q = Q_next
            # Take the average of the population
            print(P[0].fitness)
            value = [ self.f2.calculate(ind.genome) for ind in P ]
            self.state.append( max(value) )
