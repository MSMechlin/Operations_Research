import random
from Fitness import fitness_1,fitness_2
from math import floor
from Debugging import *
from sys import exit
#2 parent, 2 child crossover function using partially mapped crossover

def shuffle(shuffled):
    count = len(shuffled)
    for i in range(count):
        gen = floor(random.uniform(i,count))
        temp = shuffled[i]
        shuffled[i] = shuffled[gen]
        shuffled[gen] = temp
    return shuffled

def generate_population(problem,populationSize):
    TaskCount = Task_Count_Struct(problem)
    size = problem.size
    permutation = [*range(size)]
    population = []
    for i in range(populationSize):
        task_permutation = shuffle(permutation)
        job_permutation = problem.task_to_job_permutation(task_permutation)
        newOrganism = Organism(job_permutation,fitness_1(problem,job_permutation))
        assert Tasks_In_Jobs_Assert(TaskCount,newOrganism)
        population.append(newOrganism)
    return population

class Organism:
    def __init__(self,sequence,fitness):
        self.sequence = sequence
        self.fitness = fitness
    def __lt__(self,other):
        return self.fitness > other.fitness

def crossover(problem,parent1,parent2):
    child1,child2 = _crossover(problem,parent1.sequence,parent2.sequence)
    """
        except Exception as e:
        print("Exception Thrown")
        if e.args[0] == "Task Incomplete":
            Write_Assert_Object(e.args[1],"Crossover_Child_Task_Lost.json")
            exit()
        elif e.args[0] == "Task Repeated":
            Write_Assert_Object(e.args[1],"Crossover_Child_Task_Repeated.json")
            exit()
    """
    return Organism(child1,fitness_1(problem,child1)),Organism(child2,fitness_1(problem,child2))

def _crossover(problem, Parent1, Parent2, i = None,j = None):
    numCities = len(Parent1)    
    parent1 = problem.job_to_task_permutation(Parent1)
    parent2 = problem.job_to_task_permutation(Parent2)
    length = len(parent1)
    parent1map = [0] * length
    parent2map = [0] * length
    for t in range(length): #Set up array that contains the positions of each task id in their respective arrays. Allows for O(1) access to elements.
        parent1map[parent1[t]] = t
        parent2map[parent2[t]] = t
    child1 = []
    child2 = []
    #create a list of length numCities filled with -1's
    for l in range(numCities):
        child1.append(-1)
    child2 = child1.copy()

    #create a random segment of length numCities/2, which can wrap-around the list
    if i == None or j == None:
        i = random.randrange(0, numCities)
        j = int((i + numCities/2) % numCities)
    args = CrossoverInput(problem.problemfile,Parent1,Parent2,i,j)
    #if the segment does a wrap-around
    if j < i:
        child1[i:] = parent1[i:]
        child1[:j] = parent1[:j]
        for city in (parent2[i:] + parent2[:j]):
            if city in (parent1[i:] + parent1[:j]):
                continue
            child1 = search(parent2, child1, city, parent2map)
        for k in range(len(parent2)):
            if child1[k] == -1:
                child1[k] = parent2[k]

        child2[i:] = parent2[i:]
        child2[:j] = parent2[:j]
        for city in (parent1[i:] + parent1[:j]):
            if city in (parent2[i:] + parent2[:j]):
                continue
            child2 = search(parent1, child2, city, parent1map)
        for k in range(len(parent1)):
            if child2[k] == -1:
                child2[k] = parent1[k]

    #if the segment is in the middle of the list
    else:
        #copy the segment from parent1 into the child1 map
        child1[i:j] = parent1[i:j]
        #for each city under the segment in parent2
        for city in parent2[i:j]:
            #if the city is already copied in child1, then skip to the next city
            if city in parent1[i:j]:
                continue
            #else find the unoccupied spot where it belongs
            if city == 0:
                pass
            child1 = search(parent2, child1, city, parent2map)
        #for the remaining open spaces in child1
        for k in range(len(parent2)):
            if child1[k] == -1:
                #copy the city at the same index in parent2
                child1[k] = parent2[k]

        #form child2 using the same methods as child1, but switching the parents' places
        child2[i:j] = parent2[i:j]
        for city in parent1[i:j]:
            if city in parent2[i:j]:
                continue
            child2 = search(parent1, child2, city, parent1map)
        for k in range(len(parent1)):
            if child2[k] == -1:
                child2[k] = parent1[k]
    try:
        assert Task_Complete_Assert(numCities,child1), Exception("Task Incomplete",args)
        assert Task_Complete_Assert(numCities,child2), Exception("Task Incomplete",args)
    except Exception as e:
        raise Exception("Task Repeated",args)
    sequence1 = problem.task_to_job_permutation(child1)
    sequence2 = problem.task_to_job_permutation(child2)
    Child1 = sequence1
    Child2 = sequence2
    return Child1, Child2

#has a chance to mutate a map
#the route taken is the same, apart from the intended mutation, but the order of the list is different
def mutate(problem,child):
    sequence = child.sequence
    mutatedMap = []
    #choose a random index in map
    i = random.randrange(0, len(sequence))
    j = random.randrange(0, len(sequence))
    #if the segment doesn't wrap-around
    if i < j:
        mutatedMap = sequence[i:j]
        mutatedMap.insert(0, mutatedMap.pop())
        mutatedMap += sequence[j:] + sequence[:i]
    #if the segment does wrap-around
    elif i > j:
        mutatedMap = sequence[i:] + sequence[:j]
        mutatedMap.insert(0, mutatedMap.pop())
        mutatedMap += sequence[j:i]
    #if i == j
    else:
        mutatedMap = sequence
    return Organism(mutatedMap,fitness_1(problem,mutatedMap))
    

#replaces the bottom half of the generation sorted by fitness, repeats n/25 times before finishing
#n is the number of generations in total

#recursive function that finds the corresponding spot that is unoccupied
def search(parent2, child, city,parentmap,start=-1):
    if start == -1:
        start = city
    orig = city
    corr = child[parentmap[orig]]
    #if the corresponding city is under the data segment in Parent 2
    if child[parentmap[corr]] != -1:
        child = search(parent2, child, corr, parentmap, start)
    #if the corresponding city is not under the data segment in Parent 2
    else:
        child[parentmap[corr]] = start
    return child

class Task_Borda_Score:
    def __init__(self,id):
        self.id = id
        self.score = 0
    def __lt__(self,other):
        return self.score < other.score

def aggregate(problem,population):
    pointCounts = []
    n = len(population[0].sequence)
    for i in range(n):
        pointCounts.append(Task_Borda_Score(i))
    for individual in population:
        taskIterators = problem.jobSet.Task_Iterators()
        i = 0
        for jobCode in individual.sequence:
            task = next(taskIterators[jobCode-1])
            pointCounts[task.taskID].score += n - i
            i += 1
    pointCounts.sort()
    sequence = []
    for score in pointCounts:
        sequence.append(score.id)
    sequence = problem.task_to_job_permutation(sequence)
    time, blocks = fitness_2(problem,sequence,True)
    return Organism(sequence,time), blocks