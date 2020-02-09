def CrowdingDist(fitness=None):
    """

    :param fitness: A list of fitness values
    :return: A list of crowding distances of chrmosomes
    """

    # initialize list: [0.0, 0.0, 0.0, ...]
    distances = [0.0] * len(fitness)
    crowd = [(f_value, i) for i, f_value in enumerate(fitness)] 

    n_obj = len(fitness[0])

    for i in range(n_obj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("Inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    return distances
