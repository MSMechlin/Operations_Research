from data_structures import closedPath
from data_structures import openPath
from scipy.special import betaincinv
from data_structures import child_health
from math import inf

PARAMETER_A = 1
PARAMETER_B = 3
#Iterator returns all nodes start to end inclusive.
class start_to_end:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def __iter__(self):
        self.current = self.start
        return self
    def __next__(self):
        if self.current == self.end.child:
            raise StopIteration
        ret  = self.current
        self.current = self.current.child
        return ret

class aggreg_edge:
    def __init__(self,nextNode):
        self.next = nextNode
        self.agreement = 1  

class aggreg_node:
    def __init__(self,id):
        self.id = id
        self.outgoing = {}

class aggregation_matrix:
    def __init__(self,population):
        self.nodes = []
        for i in range(population[0].chromosome.edgecount):
            self.nodes.append(aggreg_node(i))
        for individual in population:
            path = closedPath(individual.chromosome)
            prev = next(path)
            for cur in path:
                if cur.id in self.nodes[prev.id].outgoing:
                    self.nodes[prev.id].outgoing[cur.id].agreement += 1
                else:
                    self.nodes[prev.id].outgoing[cur.id] = aggreg_edge(cur)
                prev = cur
        for node in self.nodes:
            for out in node.outgoing:
                edge = node.outgoing[out]
                edge.cost = 1 - betaincinv(PARAMETER_A,PARAMETER_B,edge.agreement/len(population))

    def cost(self,node1,node2):
        if node2.id not in self.nodes[node1.id].outgoing:
            return 1
        else:
            return self.nodes[node1.id].outgoing[node2.id].cost

def config_beta(a,b):
    global PARAMETER_A
    PARAMETER_A = a
    global PARAMETER_B
    PARAMETER_B = b

def init_ag_matrix(population):
    global matrix 
    matrix = aggregation_matrix(population)

def total_cost(path):
    cost = 0
    for current in openPath(path):
        cost += matrix.cost(current,current.child)
    return cost

def assign_cost(population,experts):
    init_ag_matrix(experts)
    for individual in population:
        individual.chromosome.cost = total_cost(individual.chromosome)

def _assign_cost(population):
    for individual in population:
        individual.chromosome.cost = total_cost(individual.chromosome)

def most_common(population):
    lowestcost = inf
    for individual in population:
        currentcost = total_cost(individual.chromosome) 
        if currentcost < lowestcost:
           lowestcost = currentcost
           lowestcostpath = individual
    return lowestcostpath

def aggregate(population):
    init_ag_matrix(population)
    _assign_cost(population)
    best = population[len(population)-1]
    best.chromosome.recalculatedist()
    print("Best path distance: {}".format(best.chromosome.length))
    total_cost(best.chromosome)
    return three_opt(best.chromosome)

def best_move(out1,out2,out3):
    removed_total = (matrix.cost(out1,out1.child) + matrix.cost(out2,out2.child) + matrix.cost(out3,out3.child))
    best = (0,0)#gain and code
    currentgain = removed_total - (matrix.cost(out1,out2.child) + matrix.cost(out2,out3.child) + matrix.cost(out3,out1.child))
    if currentgain > best[0] and currentgain > 10e-8:
        best = (currentgain,2)
    return best

def three_opt(path):
    start  = path.start
    end = path.end
    better = True
    while better:
        end1 = path.end.parent.parent.parent.parent
        end2 = path.end.parent.parent.parent
        end3 = path.end.parent
        better = False
        for current1 in start_to_end(start,end1):
            for current2 in start_to_end(current1.child.child, end2):
                for current3 in start_to_end(current2.child.child, end3):
                    move = best_move(current1,current2,current3) 
                    if move[0] > 0:
                        better = True
                        path.cost -= move[0]
                        path.reconfig(current1,current2,current3,move[1]) #Makes the improvements to the path
                        break
                if better:
                    break
            if better:
                break
            """
                Once current1 has passed start the edge pointing into start can now be removed.
            """
            end2 = path.end.parent.parent
            end3 = path.end
        path.resetend()
    path.recalculatedist()
    return path