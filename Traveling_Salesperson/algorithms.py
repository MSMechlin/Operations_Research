from math import floor,log10,inf,e
from random import uniform, random
import graphics_commands
from data_structures import *



NUMBER_OF_CYCLES = 2
MUTATIONS_PER_CHROMOSOME = 5 #Per chromosome, how many mutations on average
currentposition = 0 #Simulation of Poisson process.
processstart = False

"""
    If mutation rate is l mutations per chromosome and the distance over the chromosome that is not mutated is t then, according to the Poisson distribution where k=0, the probability of no mutations occuring in the span of t is:
    P(t) = e^-lt. 
    Therefore if we have some random number r which corresponds to the cumulative probability of no mutations occuring in the span of t then t in terms of r
    would be:
    t = -ln(r)/l
    One chromosome would be t = 1. On average  
"""

def mutate_default(path):
    global processstart
    if path.mutatearray == None:
        path.mutate_arrayInit()
    global currentposition
    brokenedges = [(path.start,path.start.child)]
    previousbroken = -1
    currentbroken = -1
    if processstart:
        previousbroken = floor((PROBLEM_SIZE-1)*currentposition)
        brokenedges.append((path.mutatearray[previousbroken],path.mutatearray[previousbroken].child))
    else:
        processstart = True
    while True:
        r = random()
        currentposition += -(log10(r) / log10(e))/(MUTATIONS_PER_CHROMOSOME)
        if currentposition >= 1:
            break
        currentbroken = floor((PROBLEM_SIZE-1)*currentposition)
        if  currentbroken != previousbroken: #An edge cannot be broken more than once. So if we end up breaking the same edge again it makes no difference.
            brokenedges.append((path.mutatearray[currentbroken],path.mutatearray[currentbroken].child)) #Parent, child tuples are required as the old child of every edge needs to be remembered
        previousbroken = currentbroken
    currentposition %= 1
    countbroken = len(brokenedges)
    permutation.reset_subset(countbroken) #Reset 0 through countbroken 
    permutation.shuffle(countbroken) #Shuffle 0 through countbroken
    if countbroken == 0:
        return path
    previous = permutation.deck[0]
    flip = True
    for i in range(1,countbroken): #N broken nodes
        current = permutation.deck[i]
        if flip:
            cur = brokenedges[current][0]
            seek = brokenedges[current-1 if current != 0 else countbroken-1][1]
            brokenedges[previous][0].child = cur
            while True:
                temp = cur.parent
                cur.parent = cur.child
                cur.child = temp
                if cur == seek:
                    break
                cur = cur.child
            flip = False
        else:
            brokenedges[previous-1 if previous != 0 else countbroken-1][1].child = brokenedges[current-1 if current != 0 else countbroken-1][1]
            brokenedges[current-1 if current != 0 else countbroken-1][1].parent = brokenedges[previous-1 if previous != 0 else countbroken-1][1]
            flip = True
        previous = current
    current = permutation.deck[0]
    if flip:
        brokenedges[previous][0].child = brokenedges[current-1 if current != 0 else countbroken-1][1]
        brokenedges[current-1 if current != 0 else countbroken-1][1].parent = brokenedges[previous][0]
    else:
        brokenedges[previous-1 if previous != 0 else countbroken-1][1].child = brokenedges[current-1 if current != 0 else countbroken-1][1]
        brokenedges[current-1 if current != 0 else countbroken-1][1].parent = brokenedges[previous-1 if previous != 0 else countbroken-1][1]
    return path

"""
    Track which graph is being traversed A or B
    Change the current vertex
    Track the edges that have been traversed
    Track vertices that could close the loop
    Store the edges in the AB_Path
    Ensure the right edge is being traversed
"""

def crossover_default(A,B):
    
    if A.edgecount != B.edgecount:
        raise Exception("The two paths have different edge counts")
    permutation.shuffle(A.edgecount)
    overlap(A,B) #Set up Gab
    cycles = []
    cur = Gab[permutation.get_current()]
    AB_path = edgeList(cur)
    while True:
        if cur.on_path:#PART OF THE PATH IS REDISCOVERED.
            backup = AB_path.back_up(cur)#Return the edge that leads out of cur on the current AB_path
            if backup.A != AB_path.end.A:
                cycles.append(AB_path.cut_cycle(backup))
            else:
                if cur == AB_path.startnode: #This is a rare edge case where a false close happens on the start of the AB_path
                    """
                    We have to remember that the AB_path is a double loop because, when we encounter the start of the path again and backtrack,
                    we do not want the new loop to start at the false close as that will make a a path that starts and ends on edges from the same graph.
                    If, instead, we run into another valid closing node and this "false close" is part of the new AB_cycle's path then "AB_path" will no longer be a
                    double loop and will be marked as such when the nodes in the the new path are unmarked.
                    """
                    AB_path.doubleloop = True
                cur.on_path = False #This way we don't trigger the if statement infinitely. Will be set back to True when pass_through is run on it again.
        else:
            """
                Only allows traversal from a node to one of its edges to find another node. Jumping could be part of this function but, 
                to keep things from geting confusing, we make a method with as simple name that does one job. This also makes testing easier as
                when passing over these functions we can see how the flow of the program communicates whether a jump or a pass through occurs.
            """
            cur = AB_path.pass_through(cur)
            """
                If we cannot find a new edge to add leading out of cur, we jump and start a new path. This algorithm guarantees that jumping will only ever happen
                when a new cycle has been cut and when that new cycle contains all of the edges of the path. All nodes in a cycle have even degrees and thus removing
                a cycle from a graph where all edges have even degrees will result in another graph with nodes of even degree. Furthermore AB_path will always have
                two nodes of odd degree.
            """
            if cur == None:
                if permutation.current < A.edgecount: 
                    cur = Gab[permutation.get_current()]
                    AB_path.startnode = cur #We are starting a completely new path from a new starting node.
                else:
                    break

    #graphics_commands.cycle_cover(closedPath(A),closedPath(B),cycles,edge_cycle_to_nodes,"Cycle Cover")

    permutation.reset_subset(len(cycles))    
    for i in range(min(NUMBER_OF_CYCLES,len(cycles))):
        current_cycle = cycles[permutation.next()]
        for edge in openPath(current_cycle):
            edge.taken = False
    
    permutation.shuffle(PROBLEM_SIZE)
    current_node = Gab[permutation.next()]
    subtours = []
    newpath = path()

    

    while True:
        open = -1
        for i in range(4):
            if valid_edge(current_node.edges[i]):
                open = i
                break
        if open != -1:
            edge = current_node.edges[i]
            edge.taken = not edge.taken
            newpath.append(pathNode(current_node.id))
            current_node = edge.opposite(current_node)
            permutation.mark(current_node.id)
        else:
            if newpath.start != None:
                newpath.close_loop()
                subtours.append(newpath)
                newpath = path()
            if permutation.current < A.edgecount:
                current_node = Gab[permutation.next()]
            else:
                break

    #graphics_commands.draw_paths(subtours,closedPath,"Subtours")

    subtours.sort()
    while True:
        if len(subtours) == 1:
            break
        bestedges = (None,None,None,None,-1,inf)
        for i in range(1,len(subtours)):
            path1 = iter(closedPath(subtours[0]))
            prev1 = next(path1)
            j = 0
            for cur1 in path1:
                path2 = iter(closedPath(subtours[i]))
                prev2 = next(path2)
                k = 0
                for cur2 in path2:         
                    dist = cur1.distance(cur2) + prev1.distance(prev2) - cur1.distance(prev1) - cur2.distance(prev2)
                    if dist < bestedges[5]:
                        bestedges = (prev2,prev1,cur2,cur1,i,dist) #Same number is the same side
                    prev2 = cur2
                    k += 1
                    if k > PROBLEM_SIZE:
                        return None
                prev1 = cur1
                j += 1
                if j > PROBLEM_SIZE:
                    return None
        if bestedges[0] == bestedges[2].child:
            bestedges[2].child = bestedges[3]
            if bestedges[1] == bestedges[3].child:
                reverse_until_end(bestedges[3],bestedges[1])
            bestedges[3].parent = bestedges[2]
            bestedges[1].child = bestedges[0]
            bestedges[0].parent = bestedges[1].parent
        else:
            bestedges[2].parent = bestedges[3]
            if bestedges[1] == bestedges[3].parent:
                reverse_until_end_backwards(bestedges[3],bestedges[1])
            bestedges[3].child = bestedges[2]
            bestedges[1].parent = bestedges[0]
            bestedges[0].child = bestedges[1]            
        subtours[bestedges[4]].edgecount = subtours[bestedges[4]].edgecount + subtours[0].edgecount
        subtours[bestedges[4]].length += subtours[0].length + bestedges[5]
        subtours[bestedges[4]].resetend()
        subtours.remove(subtours[0])
        subtours.sort()

    #graphics_commands.draw_path(closedPath(subtours[0]),"Final Result")
    return subtours[0]

def reverse_until_end_backwards(start,end):
    current = start
    while True:
        temp = current.child
        current.child = current.parent
        current.parent = temp
        if current == end:
            break
        current = current.parent

def reverse_until_end(start,end):
    current = start
    while True:
        temp = current.child
        current.child = current.parent
        current.parent = temp
        if current == end:
            break
        current = current.child

def flip_cycle(start):
    end = start.parent
    while True:
        temp = current.child
        current.child = current.parent
        current.parent = temp
        if current == end:
            break
        current = current.parent

def array_to_path(array):
    newpath = path()
    for node in array:
        newpath.append(pathNode(node))
    newpath.close_loop()
    return newpath

def valid_edge(edge):
    return (edge.A and edge.taken) or (not edge.A and not edge.taken)

"""
    0 A child
    1 A parent
    2 B child
    3 B parent
"""

def overlap(A,B): 
    apath = iter(closedPath(A))
    prev = next(apath) 
    for cur in apath:
        edge = Gab[prev.id].edges[0]
        edge.parent_id = prev.id
        edge.child_id = cur.id
        edge.taken = False
        edge.parent = None
        edge.child = None
        Gab[cur.id].edges[1] = edge
        prev = cur
    
    bpath = iter(closedPath(B))
    prev = next(bpath) 
    for cur in bpath:
        edge = Gab[prev.id].edges[2]
        edge.parent_id = prev.id
        edge.child_id = cur.id
        edge.taken = False
        Gab[cur.id].edges[3] = edge
        prev = cur



def child_health(child,expected_weight_of_the_newborn_infant):
    permutation.reset_subset(expected_weight_of_the_newborn_infant)
    i = 0 
    for current in openPath(child):
        #print("{}: {},".format(i,current.id))
        permutation.mark(current.id)
        i+=1
        if i>PROBLEM_SIZE:
            return False
    if permutation.current != expected_weight_of_the_newborn_infant:
        return False
    return True