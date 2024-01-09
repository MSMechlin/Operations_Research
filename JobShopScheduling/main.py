from random import random
from Fitness import Problem, fitness_2
from time import time
from functions import crossover, mutate, Organism
from functions import generate_population, aggregate
from Debugging import *
import Graphics
# Graphics.init_graphics()
import math
from functools import reduce
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
    Selection is carried out using a ranking scheme. The probability of selection is entirely dependent on 
    rank and not "fitness". In fact fitness is really not used to compare individuals. The individuals are
    just ranked by how short their paths are. 
"""

def less_than(self,other):
    return self.fitness[0] < other.fitness[0] or (self.fitness[0] == other.fitness[0] and self.fitness[1] < other.fitness[1])

def select(population):
    r = random()
    k = math.ceil((-2*ALPHA - BETA + math.sqrt((2*ALPHA + BETA)**2 + 8*BETA*r))/(2*BETA))
    return population[k-1]

def data_point(population): #If the list is sorted, the most fit should be the last in the list.
    return population[len(population)-1].objective 

def save_population(popuplation,filename):
    file = open(filename,"w")
    for individual in popuplation:
        nodes = iter(individual)
        node = next(nodes)
        file.write("{}".format(node.id))
        for node in nodes:
            file.write(",{}".format(node.id))
        file.write("\n")
    file.close()

"""
def load_population(filename):
    file = open(filename,"r")
    population = []
    for line in file:
        order = line.split(",")
        newpath = #CONTAINER
        for node in order:
            #newpath.append(int(node))
        #population.append(newpath)
    file.close()
    return population
"""

data = []
def genetic_algorithm(problem,population, points,gen_limit = None):
    countStruct = Task_Count_Struct(problem)
    currentGeneration = 1
    global data
    if gen_limit == None:
        gen_limit =  GENERATION_LIMIT
    while currentGeneration <= gen_limit:
        population.sort()

        ##PLOT BEST IN GEN
        points.append(population[-1].fitness)

        newpopulation = []
        for i in range(len(population)//2 - 1): 
            x = select(population)
            y = select(population)
            child1,child2 = crossover(problem,x,y)
            if random() <= MUTATION_RATE:
                child1 = mutate(problem,child1)
            if random() <= MUTATION_RATE:
                child2 = mutate(problem,child2)
            newpopulation.append(child1)
            newpopulation.append(child2)

        ##KEEP TOP 2
        newpopulation.append(population[-1])
        # newpopulation.append(population[-2])
        # newpopulation.append(population[-3])

        ##WOC
        agg1, blocks = aggregate(problem,population)

        newpopulation.append(agg1)
        # newpopulation.append(agg2)


        population = newpopulation
        currentGeneration += 1

    population.sort()
    return population

import os
filepath = os.path.join(os.path.dirname(__file__), "Problem3.jsh")
prob = Problem(filepath)
POPULATION_SIZE = 100
i = 1
pop = generate_population(prob,POPULATION_SIZE)

print("Best from population")
best, blocks = fitness_2(prob,pop[len(pop)-1].sequence,True)
# Graphics.draw_gantt(blocks,best,prob.resourceCount)

MUTATION_RATE = .05

GENERATION_LIMIT = 500
SELECTION_PRESSURE = 1.5 #The ratio of the probability of the hottest chromosome getting some action over the probability of the average joe settling down.
ALPHA = (2*POPULATION_SIZE - SELECTION_PRESSURE*(POPULATION_SIZE+1))/(POPULATION_SIZE*(POPULATION_SIZE-1))
BETA = (2*(SELECTION_PRESSURE-1))/(POPULATION_SIZE*(POPULATION_SIZE-1))
points = []
ret = genetic_algorithm(prob,pop, points, GENERATION_LIMIT)

##PLOT POINTS
print(points)
plt.plot(points)
plt.show()

# print("Aggregate solution")
# agg, blocks = aggregate(prob,ret)
# print(agg.sequence, agg.fitness)

# Graphics.draw_gantt(blocks,agg.fitness,prob.resourceCount)