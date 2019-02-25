#!/user/bin/env python3

import argparse
from itertools import combinations, chain
from operator import itemgetter
from os.path import isfile
from sys import argv
from multiprocessing import Pool, cpu_count
from re import match
from random import randint, choice, sample, random
from typing import List, Tuple, Callable

Tour = List[int]
TourCost = Tuple[Tour, int]
DistanceMatrix = List[List[int]]

cf_re = r"^NAME=(?P<name>.*),\nSIZE=(?P<size>\d*),\nDISTANCES=(?P<distances>(\d*,?)*)$"
def read_city_file(file_address: str)-> Tuple[str, int, List[List[int]]]:
    # read raw data in file
    with open(file_address, "r") as f:
        raw = f.read()

    # try and parse data
    data = match(cf_re, raw)
    assert data != None, "Badly formatted file, expecting format matching re: {}".format(cf_re)
    name, size, distances = (data.group("name"), int(data.group("size")),
                            [int(int_str) for int_str in data.group("distances").split(",")])
    assert len(distances) == 0.5*size*(size-1), "Wrong number of distances in file: {}".format(len(distances))

    # build distance matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for u, v in combinations(range(size), 2):
        matrix[u][v] = distances.pop(0)
        matrix[v][u] = matrix[u][v]

    return name, size, matrix

def create_tsp(name: str, size: int, filename: str, max_dist=500):
    distances_string = ",".join(str(randint(0, 500)) for i in range(int(0.5*size*(size-1))))
    with open(filename, "w+") as f:
        f.write("NAME={},\nSIZE={},\nDISTANCES={}".format(name, size, distances_string))

def tour_cost(dist_matrix: DistanceMatrix, tour: Tour, size: int) -> int:
   return sum(dist_matrix[tour[i]][tour[(i-1) % len(tour)]] for i in range(len(tour)))

def mutate_swap(tour: Tour, probability: float, size: int) -> List[int]:
    if random() <= probability:
        i, j = (randint(0, size - 1) for _ in range(2))
        tour[i], tour[j] = tour[j], tour[i]
        return mutate_swap(tour, probability, size)
    return tour

def mutate_reverse(tour: List[int], probability: float, size: int) -> List[int]:
    if random() <= probability:
        i, j = sorted(randint(0, size - 1) for _ in range(2))
        tour[i:j] = reversed(tour[i:j])
    return tour

def cx_partial_map(p1: List[int], p2: List[int], size: int) -> Tuple[List[int], List[int]]:
    cxp = randint(0, size - 1)
    c1, c2 = p1.copy(), p2.copy()
    for i in range(cxp + 1):
        t1, t2 = c1[i], c2[i] # displaced values
        c1[i], c2[i] = p2[i], p1[i]
        for j in range(i + 1, size):
            if c1[j] == p2[i]: c1[j] = t1
            if c2[j] == p1[i]: c2[j] = t2
    return c1, c2

def sl_tournament(n: int, pool: List[Tuple[List[int], int]]) -> List[Tuple[List[int], int]]:
    parents = pool.copy()
    while len(parents) > n:
        i, j = (randint(0, len(parents) - 1) for _ in range(2))
        parents.pop(j) if parents[i][1] < parents[j][1] else parents.pop(i)
    return parents

def threaded_genetic(dist_matrix: List[List[int]], size: int, population_size: int, generations: int,
            mutation_probability: float, elite_offset: int, disqualification_offset: int, parent_pool_size: int,
            mutator: Callable[[List[int], float, int], List[int]],
            crossover: Callable[[List[int], List[int], int], Tuple[List[int], List[int]]],
            selection: Callable[[int, List[Tuple[List[int], int]]], List[Tuple[List[int], int]]]):
    workers = Pool(cpu_count())

    #build inital population
    cities = list(range(size))
    print("Building initial population...")
    pop = workers.starmap(local_search, ((dist_matrix, sample(cities, size), size) for c_num in range(population_size)))

    #do generations
    for g_num in range(generations):
        print("Current population of size {} have an average fitness of {}".format(population_size, sum(cost for (tour, cost) in pop)/population_size))
        print("Generation {}/{}".format(g_num + 1, generations))
        pop.sort(key=lambda pair: pair[1])
        pop, parent_pool = pop[:elite_offset], selection(parent_pool_size, pop[:-disqualification_offset])
        children = []
        while len(children) < (population_size - elite_offset):
            children += [mutator(c, mutation_probability, size) for c in crossover(choice(parent_pool)[0],
                                                                                   choice(parent_pool)[0], size)]
        pop += workers.starmap(local_search, ((dist_matrix, c, size) for c in children))

    return min(pop, key=lambda pair: pair[1])

def local_search(dist_matrix: List[List[int]], tour: List[int], size: int) -> Tuple[List[int], int]:
        cost, init_cost = [tour_cost(dist_matrix, tour, size)]*2
        while True:
            improvement_made = False
            for i, j in combinations(range(size), 2):
                diff = score_diff(dist_matrix, tour, size, i, j)
                if diff < 0 :
                    cost += diff
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improvement_made = True
            if not improvement_made: break
        print("Initial cost: {}... Improved cost: {}".format(init_cost, cost))
        return tour, cost

def threaded_local_search(dist_matrix: DistanceMatrix, tour: Tour, size: int, workers: Pool, threads):
    cost = tour_cost(dist_matrix, tour, size)
    pairs = combinations(range(size), 2)
    while True:
        diff, (i, j) = min(zip(workers.starmap(score_diff,
                                               ((dist_matrix, tour, size, i, j) for (i, j) in pairs)), pairs),
                           key=itemgetter(0))
        if diff >= 0: break
        else:
            cost += diff
            tour[i:j + 1] = reversed(tour[i:j + 1])
    return tour, cost

def score_diff(dist_matrix: DistanceMatrix, tour: Tour, size: int, i: int, j: int):
    if (i, j) == (0, size - 1):
        return 0
    else:
        u_prev, u, v, v_next = tour[(i - 1) % size], tour[i], tour[j], tour[(j + 1) % size]
        return dist_matrix[u_prev][v] + dist_matrix[u][v_next] - dist_matrix[u_prev][u] - dist_matrix[v][v_next]

def solve(city_file: str, output_file: str):
    name, size, dist_matrix = read_city_file(city_file)
    tour, cost = threaded_genetic(dist_matrix, size, 128, 128, 0.15, 16, 32, 96, mutate_swap, cx_partial_map, sl_tournament)
    print("Best cost: {}".format(cost))
    with open(output_file, "w+") as f:
        f.write("NAME = {},\nTOURSIZE = {},\nLENGTH = {},\nTOUR={}".format(
        name, size, cost, ",".join(str(city) for city in tour)))


def cmd_create(args):
    create_tsp(args.name, args.size, args.filename, args.maximum)

def cmd_solve(args):
    name, size, dist_matrix = read_city_file(args.problem_file)
    tour, cost = threaded_genetic(dist_matrix, size, 256, 256, 0.1, 32, 8, 160, mutate_swap, cx_partial_map, sl_tournament)
    print("Best cost: {}".format(cost))
    with open(args.output_file, "w+") as f:
        f.write("NAME={},\nTOURSIZE={},\nLENGTH={},\nTOUR={}".format(name, size, cost, ",".join(str(city) for city in tour)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program contains algorithms for solving the tsp")
    commands = parser.add_subparsers()
    create = commands.add_parser("create", help="Create a new random tsp instance and save it to file")
    create.add_argument("name", help="Name of problem")
    create.add_argument("size", type=int, help="Number of cities in problem")
    create.add_argument("filename", help="File address to save to")
    create.add_argument("-maximum", "-m", type=int, default=500,
    help="Maximum distance between two cities")
    create.set_defaults(func=cmd_create)

    solve = commands.add_parser("solve", help="Solve a tsp instance")
    solve.add_argument("problem_file", help="File address of file containing problem")
    solve.add_argument("output_file", help="File address to be written to")
    solve.set_defaults(func=cmd_solve)
    args = parser.parse_args()
    args.func(args)