from Debugging import *
from functions import _crossover
from Fitness import Problem

problem = Problem("Problem1.jsh")
assertObj = Load_Assert_Object(CrossoverInput,"Crossover_Child_Task_Repeated.json")
_crossover(problem,assertObj.parent1,assertObj.parent2,assertObj.i,assertObj.j)