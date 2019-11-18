import numpy as np



def tournament(P, number_to_select):
    ''' Tournament selection
    args:
        - P: Poblation, a list of Individuals
        - number_of_contestant: Integer
    '''

    individuals = []
    for _ in number_to_select:
        y = P[np.random.randint(0, len(P) -  1)]
        for _ in range(3):
            z = P[np.random.randint(0, len(P) - 1)]
            y = z if z.fitness < y.fitness else y
        individuals.append(y)
    return individuals

