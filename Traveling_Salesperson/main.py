from random import random
from data_structures import *
#init_data_structures("Random22.tsp")
from graphics_commands import display_saved_surface
from Aggregation import aggregate, config_beta, assign_cost
from data_structures import openPath
from time import time
import re

import math
import matplotlib.pyplot as plt

import scipy.stats as stats


MUTATION_RATE = .05
POPULATION_SIZE = 100
GENERATION_LIMIT = 50
SELECTION_PRESSURE = 1.5 #The ratio of the probability of the hottest chromosome getting some action over the probability of the average joe settling down.
ALPHA = (2*POPULATION_SIZE - SELECTION_PRESSURE*(POPULATION_SIZE+1))/(POPULATION_SIZE*(POPULATION_SIZE-1))
BETA = (2*(SELECTION_PRESSURE-1))/(POPULATION_SIZE*(POPULATION_SIZE-1))

def init_histogram():
    global histogram
    histogram = []
    for i in range(POPULATION_SIZE):
        histogram.append(0)

class organism:
    def __init__(self,ch):
        if ch == None:
            return None
        self.chromosome = ch
        self.objective = objective(self.chromosome)
    def __lt__(self,other): #list.sort() only needs the less than operator.
        return self.objective > other.objective #LOOK OUT RANKING MIGHT BE INVERTED MAKE SURE k = 100 has highest probability
"""
    Selection is carried out using a ranking scheme. The probability of selection is entirely dependent on 
    rank and not "fitness". In fact fitness is really not used to compare individuals. The individuals are
    just ranked by how short their paths are. 
"""
def selection_default(population):
    r = random()
    k = math.ceil((-2*ALPHA - BETA + math.sqrt((2*ALPHA + BETA)**2 + 8*BETA*r))/(2*BETA))
    return population[k-1]

def data_point(population): #If the list is sorted, the most fit should be the last in the list.
    return population[len(population)-1].objective 

def objective_default(individual):
    return individual.length

objective = objective_default
select = selection_default
crossover = crossover_default
mutate = mutate_default


def generate_population(size):
    population = []
    for i in range(size):
        population.append(organism(generate_path()))
    return population

def length_generator(pop):
    for path in pop:
        yield path.chromosome.length

def save_population(pop,filename):
    file = open(filename,"w")
    for individual in pop:
        nodes = iter(openPath(individual.chromosome))
        node = next(nodes)
        file.write("{}".format(node.id))
        for node in nodes:
            file.write(",{}".format(node.id))
        file.write("\n")
    file.close()

def load_population(filename):
    file = open(filename,"r")
    population = []
    for line in file:
        order = line.split(",")
        newpath = path()
        for node in order:
            newpath.append(pathNode(int(node)))
        newpath.close_loop()
        population.append(organism(newpath))
    file.close()
    return population

data = []
def genetic_algorithm(population,gen_limit = None):

    global currentSet
    currentGeneration = 1
    global data
    if gen_limit == None:
        gen_limit =  GENERATION_LIMIT
    while currentGeneration <= gen_limit:
        population.sort()
        #if currentGeneration%5 == 0:
            #save_population(population,"Data_Size_{}/Genetic_Algorithm_S{}_P100_G{}_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,currentGeneration,currentSet))
        #data.append(data_point(population))#Add most fit member of population.
        newpopulation = []
        for i in range(len(population)): 
            x = select(population)
            y = select(population)
            child = None
            #I haven't worked out all of the bugs, there is about a 7% chance that the crossover algorithm will get caught in an infinite loop while generating
            #an intermediate solution. I have to abort it when the condition that causes the loop occurs.
            while True:  
                newchromo = crossover(x.chromosome,y.chromosome)
                if newchromo == None:
                    #print("Infinite Loop")
                    continue
                if child_health(newchromo,PROBLEM_SIZE):
                    newchromo.recalculatedist()
                    child = organism(newchromo)
                    break
                #print("Unhealthy child")
            if random() <= -MUTATION_RATE:
                #print("MUTATION")
                child = organism(mutate(child.chromosome))
            newpopulation.append(child)
        population = newpopulation
        #print("Current generation: {}".format(currentGeneration))
        print("Current generation completed: {}".format(currentGeneration))
        currentGeneration += 1
        
    population.sort()
    return population

print("Input file path containing problem")
filename = input()
PROBLEM_SIZE = init_data_structures(filename)
population = generate_population(POPULATION_SIZE)
population = genetic_algorithm(population)

graphics_commands.draw_path(closedPath(population[len(population)-1].chromosome),"Phase 1")
graphics_commands.display_saved_surface("Phase 1")
#RETURNS PATH NOT ORGANISM

final = aggregate(population)
final.recalculatedist()
print("Aggregate path distance: {}".format(final.length))
graphics_commands.draw_path(closedPath(final),"Result")
graphics_commands.display_saved_surface("Result")







"""
##################################################################################################################################################
#################################### CODE FOR TESTING BELOW #########################################################################
##################################################################################################################################################
"""


def generate_data(filename):
    global timedata
    #timedata = open("Timetest_unaltered_{}.txt".format(filename),"w")
    global currentSet
    global PROBLEM_SIZE
    PROBLEM_SIZE = init_data_structures(filename)
    pop = generate_population(POPULATION_SIZE)  
    for gen_limit in range(100,101,5):
        #timedata.write("Generation limit: {}\n".format(gen_limit))
        print("Generation limit: {}".format(gen_limit))
        for j in range(15):
            currentSet = j+1
            #start = time()
            save = genetic_algorithm(pop,gen_limit)
            print("Datapoint {} completed".format(j+1))
            #end = time()
            save_population(save,"Data_Size_{}/Genetic_Algorithm_S{}_P100_G{}_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,gen_limit,j+1))
            #print("Datapoint {}; Time elapsed: {}".format(j+1,end-start))
            #timedata.write("{}\n".format(end-start))
    timedata.close()


timetestfiles2 = ["Timetest_Aggregate_Random11.tsp.txt","Timetest_Aggregate_Random22.tsp.txt","Timetest_Aggregate_Random44.tsp.txt","Timetest_Aggregate_Random77.tsp.txt","Timetest_Aggregate_Random97.tsp.txt", "Timetest_Aggregate_Random222.tsp.txt"]
timetestfiles = ["Timetest_unaltered_Random11.tsp.txt","Timetest_unaltered_Random22.tsp.txt","Timetest_unaltered_Random44.tsp.txt","Timetest_unaltered_Random77.tsp.txt","Timetest_unaltered_Random97.tsp.txt", "Timetest_unaltered_Random222.tsp.txt"]
def time_bar_graph(genlimit):
    bottom = []
    names = ["n=11","n=22","n=44","n=77","n=97","n=222"]
    time_unaltered = []
    for file in timetestfiles:
        time_unaltered.append(average_time_for_genlimit(file,genlimit))
        bottom.append(0)
    time_enhancement = []
    for file in timetestfiles2:
        time_enhancement.append(average_time_for_genlimit(file,genlimit))
    fig, ax = plt.subplots()
    ax.bar(names, time_unaltered, .75, label=True, bottom=bottom)
    ax.bar(names, time_enhancement, .75, label=True, bottom=time_unaltered)
    plt.show()

def scatter_cost_length(filename):
    COST = []
    LENGTH = []
    PROBLEM_SIZE = init_data_structures(filename)
    for j in range(15):
        pop = load_population("Data_Size_{}/Genetic_Algorithm_S{}_P100_G100_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,j+1))
        assign_cost(pop,pop[90:100])
        for individual in pop:
            COST.append(individual.chromosome.cost)
            LENGTH.append(individual.chromosome.length)
        plt.scatter(COST,LENGTH)
        plt.xlabel("Cost")
        plt.ylabel("Length")
        plt.show()
        stats.pearsonr(COST,LENGTH)
        COST = []
        LENGTH = []

def r_over_generation(filename):
    average_rs = []
    generations = []
    COST = []
    LENGTH = []
    PROBLEM_SIZE = init_data_structures(filename)
    for i in  range(5,51,5):
        average_rs.append(0)
        generations.append(i)
    for i in  range(5,51,5):
        for j in range(15):
            pop = load_population("Data_Size_{}/Genetic_Algorithm_S{}_P100_G{}_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,i,j+1))
            assign_cost(pop,pop[90:100])
            for individual in pop:
                COST.append(individual.chromosome.cost)
                LENGTH.append(individual.chromosome.length)
            average_rs[(i//5)-1] += stats.pearsonr(COST,LENGTH)[0]
            COST = []
            LENGTH = []
        average_rs[(i//5)-1] /= 15
    fig, ax = plt.subplots()
    ax.plot(generations,average_rs)
    plt.xlabel("Generation")
    plt.ylabel("Average pearson correlation")
    ax.set_title("Average Person Correlation vs Generation Limit")
    plt.show()

def plot_data(filename):
    GENERATION = []
    AVERAGE_UNALTERED = []
    AVERAGE_AGGREGATED = []
    global timedata
    timedata = open("Timetest_Aggregate_{}.txt".format(filename),"w")
    for j in range(50,51,5):
        GENERATION.append(j)
    PROBLEM_SIZE = init_data_structures(filename)
    for i in range(50,51,5):
        sum_of_min_length = 0
        sum_of_aggregate = 0
        timedata.write("Generation limit: {}\n".format(i))
        print("Generation limit: {}".format(i))
        for j in range(15):
            pop = load_population("Data_Size_{}/Genetic_Algorithm_S{}_P100_G{}_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,i,j+1))
            sum_of_min_length += pop[len(pop)-1].objective
            start = time()
            altered = aggregate(pop[90:100])
            sum_of_aggregate += altered.length
            end = time()
            timedata.write("{}\n".format(end-start))
            print("Datapoint {}; Time elapsed: {}".format(j+1,start-end))       
        AVERAGE_UNALTERED.append(sum_of_min_length/15)
        AVERAGE_AGGREGATED.append(sum_of_aggregate/15)
    fig, ax = plt.subplots()
    ax.plot(GENERATION,AVERAGE_UNALTERED,"b")
    ax.plot(GENERATION,AVERAGE_AGGREGATED,"g")
    ax.set_title("UNALTERED(BLUE) VS ENHANCED(GREEN)")
    plt.xlabel("Number of generations")
    plt.ylabel("Average solution length")
    plt.show()
    timedata.close()

    

def beta_test(filename):
    PROBLEM_SIZE = init_data_structures(filename)
    for i in range(5,51,5): #FIX THIS
        sum_of_min_length = 0
        sum_of_aggregate = 0
        beta_yaxis = []
        beta_xaxis = []
        for b in range(10):
            beta_yaxis.append(0)
            beta_xaxis.append(b+1)
        for j in range(15): #FIX THIS
            pop = load_population("Data_Size_{}/Genetic_Algorithm_S{}_P100_G{}_{}.txt".format(PROBLEM_SIZE,PROBLEM_SIZE,i,j+1))
            min_unaltered = pop[len(pop)-1]
            sum_of_min_length += min_unaltered.objective
            for b in range(len(beta_yaxis)):
                config_beta((b+1)/5,(b+1)/5)
                altered = aggregate(pop[90:100])
                beta_yaxis[b] += altered.length - min_unaltered.objective
                print("Beta Tested: {}".format(b+1))
            print("Datapoint completed: {}".format(j+1))
        for b in range(len(beta_yaxis)):
            beta_yaxis[b] /= 15
        fig, ax = plt.subplots()
        ax.plot(beta_xaxis,beta_yaxis,"b")
        ax.set_title("Difference vs Beta Generation: {}".format(i))
        plt.xlabel("Beta")
        plt.ylabel("Average Difference")
        plt.show()

def average_time_for_genlimit(filename,genlimit):
    file = open(filename,"r")
    sum = 0
    count =0
    for line in file:
        if re.match("Generation limit: \d+",line):
            halves = line.split(":")
            compare = int(halves[1])
            if compare == genlimit:
                break
    for line in file:
        if re.match("Generation limit: \d+",line):
            break
        sum += float(line)
        count += 1
    return sum/count
"""
##################################################################################################################################################
#################################### CODE FOR TESTING ABOVE #########################################################################
##################################################################################################################################################
"""