import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from typing import Tuple
from matplotlib import pyplot as plt
import random
from scipy.optimize import fmin
from scipy.spatial.distance import euclidean
from itertools import product
from copy import deepcopy


POPULATION_SIZE = 100
P_CROSSOVER = 0.6       
P_MUTATION = 0.00005       
MAX_GENERATIONS = 50


def the_best_algorithm_in_the_world(polygon: Polygon, things: Tuple[Polygon]) -> bool:
    if polygon.is_empty or len(things) == 0:
        return False
    if polygon.area < sum([obj.area for obj in things]):
        return False

    tmp_polygon = Polygon(polygon)

    for thing in things:
        rot, xoff, yoff, fopt = genetic_alg(tmp_polygon, thing)
        if fopt == -np.inf:
            return False
       
        tmp_thing = rotate(translate(thing, xoff=xoff, yoff=yoff), rot)
        tmp_polygon = tmp_polygon.difference(tmp_thing)
    return True

"""
Дальше вы можете увидеть разбросанные функции, не приведённые к особо читабельному виду.
Некоторые строчки кода можно упростить, переписать более нормальным образом. Но по времени уже близился дедлайн, поэтому оставил всё, как есть
Большая часть того, что вы можете лицезреть, вероятнее всего, будет изменена координально.
"""

def distance_between_contours(polygon, thing):
    points_first = [*zip(*polygon.exterior.coords.xy)]
    points_second = [*zip(*thing.exterior.coords.xy)]
    crosses = list(product(points_first, points_second))
    distanses = list(euclidean(points[0], points[1]) for points in crosses)
    return distanses


def cnt_points(individual, value):
    cnt = list(filter(lambda x: x < value, individual.distanses))
    return len(cnt)


def get_optimal_off(polygon, thing, rot=0, xoff=0, yoff=0):
    distanses = []
    def f(args):
        rot, xoff, yoff = args
        another = rotate(translate(thing, xoff=xoff, yoff=yoff), rot)
        if not polygon.contains(another):
            return np.inf
        nonlocal distanses
        distanses = distance_between_contours(polygon.simplify(3), another.simplify(3))    
        return min(distanses)
    result = fmin(f, np.array([rot, xoff, yoff]), full_output=True, disp=False)

    
    rot, xoff, yoff = result[0]
    fopt = result[1]
    if fopt == np.inf:
        rot = xoff = yoff=  0

    return rot, xoff, yoff, fopt, distanses


class Individual():
    def __init__(self, params=[0, 0, 0]):
        self.fitness = 0
        self.params = params
        self.begin_params=[]
        self.distanses = []
        


def clone(value):
    ind = Individual(value.params[:])
    ind.fitness = value.fitness
    ind.begin_params = deepcopy(value.begin_params)
    ind.distanses = deepcopy(value.distanses)
    return ind


def selection_tournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(
                0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max(
            [population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness))

    return offspring


def individual_creator(polygon, thing):
    dst = int(polygon.hausdorff_distance(thing))
    minx, miny, maxx, maxy = map(int, polygon.bounds)
    x0, y0 = thing.centroid.coords.xy
    cnt = 0
    x0, y0 = x0[0], y0[0]
    while True:
        cnt+=1
        x1 = random.randint(minx ,maxx)
        y1 = random.randint(miny, maxy)

        rot0 = random.randint(0, 359)
        xoff0 = x1 - x0
        yoff0 = y1 - y0
        tmp = rotate(translate(thing, xoff=xoff0, yoff=yoff0), rot0)     
        if cnt > 50 or polygon.contains(tmp):
            break
   

    thing = rotate(translate(thing, xoff=xoff0, yoff=yoff0), rot0)
    rot, xoff, yoff, fopt, distanses = get_optimal_off(
        polygon, thing, 0, 0, 0)
    
    individual = Individual([rot, xoff, yoff])
    individual.begin_params = [rot0, xoff0, yoff0]
    individual.distanses = distanses
     
    return individual


def population_creator(polygon, thing, n=0):
    return [individual_creator(polygon, thing) for i in range(n)]


def cx_one_point(child1, child2):
    s = random.randint(0, 2)
    child1.params[s:], child2.params[s:] = child2.params[s:], child1.params[s:]


def mutate(mutant, polygon, thing, mutation_param=0.01):
    dst = int(polygon.hausdorff_distance(thing))
    for indx in range(len(mutant.params)):
        if random.random() < mutation_param:
            if indx == 0:
                mutant.params[indx] = 0
            elif indx == 1:
                mutant.params[indx] = 0
            else:
                mutant.params[indx] = 0
                


def max_fitness(individual, polygon, thing):
    rot = individual.params[0]
    xoff = individual.params[1]
    yoff = individual.params[2]
    rot0 ,xoff0, yoff0 = individual.begin_params
    tmp = rotate(translate(thing, xoff=xoff0, yoff=yoff0), rot0)
    tmp = rotate(translate(tmp, xoff=xoff, yoff=yoff), rot)   
    if not polygon.contains(tmp):
        return -np.inf
     
    return cnt_points(individual, 30)

def genetic_alg(polygon: Polygon, thing: Polygon):
    population = population_creator(polygon, thing, n=POPULATION_SIZE)
    generation_counter = 0

    fitness_values = [max_fitness(individual, polygon, thing) for individual in population]

    for individual, fitnessValue in zip(population, fitness_values):
        individual.fitness = fitnessValue

    fitness_values = [individual.fitness for individual in population]

    while generation_counter < MAX_GENERATIONS:
        generation_counter += 1
        offspring = selection_tournament(population, len(population))

        offspring = list(map(clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                cx_one_point(child1, child2)

        for mutant in offspring:
            if random.random() < P_MUTATION:
                mutate(mutant, polygon, thing, mutation_param=0.01)

        fresh_fitness_values = [max_fitness(
            individual, polygon, thing) for individual in offspring]
        for individual, fitnessValue in zip(offspring, fresh_fitness_values):
            individual.fitness = fitnessValue

        population[:] = offspring

        fitness_values = [ind.fitness for ind in population]

        best_index = fitness_values.index(max(fitness_values))
        print("Лучший индивидуум = ", max(fitness_values),
              *population[best_index].params, "\n")
    rot, xoff, yoff = population[best_index].params
    rot0, xoff0, yoff0 = population[best_index].begin_params
    
    return  rot + rot0, xoff + xoff0, yoff + yoff0,  population[best_index].fitness
