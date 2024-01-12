from math import floor,log10,sqrt,pow,inf,e
from random import uniform, random

def init_data_structures(filename):
    init_vertices(filename)
    init_Gab()
    permutation.init()
    return PROBLEM_SIZE

def init_vertices(filename):
    global nodearray
    nodearray = []
    nodefile = open(filename)
    nodearray = addNodes(nodearray,nodefile)
    global PROBLEM_SIZE 
    PROBLEM_SIZE = len(nodearray)
    global edgeweights
    edgeweights = [[]]
    for i in range(PROBLEM_SIZE):
        for j in range(PROBLEM_SIZE):
            edgeweights[i].append(-1)
        edgeweights.append([])

def init_Gab():
    size = PROBLEM_SIZE
    global Gab
    Gab = []
    for i in range(size):
        Gab.append(Gab_Node(i))
    for i in range(1,size):
        Gab[i-1].edges[0] = edge(i-1,i,True)
        Gab[i-1].edges[2] = edge(i-1,i,False)
        Gab[i].edges[1] = Gab[i-1].edges[0]
        Gab[i].edges[3] = Gab[i-1].edges[2]
    Gab[size-1].edges[0] = edge(size-1,0,True)
    Gab[size-1].edges[2] = edge(size-1,0,False)
    Gab[0].edges[1] = Gab[size-1].edges[0]
    Gab[0].edges[3] = Gab[size-1].edges[2]


class permutation: #Hopefully I will only need one permuation at a time.    
    def init():
        permutation.deck = [*range(PROBLEM_SIZE)]
        permutation.map = [*range(PROBLEM_SIZE)]
        current = 0
    def shuffle(count):
        for i in range(count):
            while True:
                gen = floor(uniform(i,count))
                if gen != count:#There is a very slight chance that the least upper bound of the interval [i,PROBLEM_SIZE] will be chosen.
                    break
            permutation.swap(i,gen)
        permutation.restart()
    def reset():
        permutation.deck = [range(PROBLEM_SIZE)]
    
    def restart():
        permutation.current = 0

    def swap(a,b):
        permutation.map[permutation.deck[a]] = b
        permutation.map[permutation.deck[b]] = a
        temp = permutation.deck[a]
        permutation.deck[a] = permutation.deck[b]
        permutation.deck[b] = temp     

    def mark(number):
        pos = permutation.map[number]
        cur = permutation.current
        if pos < cur:
            return False
        permutation.current += 1
        permutation.swap(cur,pos)
        return True

    def reset_subset(length): # Resets numbers 0 through length - 1 so permutations of a smaller size can be generated.  
        for i in range(length):
            permutation.swap(permutation.map[i],i)
        permutation.shuffle(length)
        permutation.current = 0
    
    def pop_current():
        permutation.current += 1
        return permutation.deck[permutation.current-1]
    
    def get_current():
        return permutation.deck[permutation.current]
        
    def next():
        ret = permutation.deck[permutation.current]
        permutation.current += 1
        return ret
    
    def current():
        return permutation.deck[permutation.current]
    
    def init_mark():
        permutation.deck = [range(PROBLEM_SIZE)]
        permutation.map = [range(PROBLEM_SIZE)]
        current = 0 

class point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __init__(self,id,x,y):
        self.id = id
        self.x = x
        self.y = y
    def distance(self,other): #This algorithm is going to be lowerbounded by O(N^2) so might as well initialize a 2d array. Certainly not worth doing sqrt operation ad hoc.
        if edgeweights[self.id][other.id] == -1:
            edgeweights[self.id][other.id] = sqrt(pow(self.x-other.x,2) + pow(self.y-other.y,2))
        return edgeweights[self.id][other.id]
    """Finds shortest distance between self and the line segment with vertices a and b"""
   #distance to line

def addNodes(nodes,nodefile):
    for line in nodefile:
        if line == "NODE_COORD_SECTION\n":
            break
    columns = []
    for line in nodefile:
        columns = line.split(" ")
        nodes.append(point(int(columns[0])-1,float(columns[1]),float(columns[2])))
    return nodes

class Gab_Node:
    def __init__(self,id):
        self.edges = [None,None,None,None]
        self.id = id
        self.on_path = False
    def getx(self):
        return nodearray[self.id].x
    def gety(self):
        return nodearray[self.id].y

class edge:
    def __init__(self,ind,outd,A):
        self.child_id = ind
        self.parent_id = outd
        self.A = A
        self.taken = False
        self.child = None
        self.parent = None
    def opposite(self,node):
        if node.id == self.parent_id:
            return Gab[self.child_id]
        else:
            return Gab[self.parent_id]

class edge_cycle_to_nodes:
    def __init__(self,edgelist):
        self.currentnode = edgelist.startnode
        self.currentedge = edgelist.start
        self.firstreturned = False
        self.finalreturned = False
        self.end = edgelist.endnode
    def __iter__(self):
        return self
    def __next__(self):
        if self.finalreturned:
            raise StopIteration
        ret = self.currentnode
        self.currentnode = self.currentedge.opposite(self.currentnode)
        self.currentedge = self.currentedge.child
        self.finalreturned = self.firstreturned and ret == self.end
        self.firstreturned = True
        return ret

class edgeList:
    def __init__(self,startnode,endnode=None,startedge=None,endedge=None):
        self.start = startedge
        self.end = endedge
        if startnode == endnode and self.start != None: #Thsis is the special case for initializing a loop
            self.start.parent = self.end
            self.end.child = self.start
        self.startnode = startnode
        self.endnode = endnode
        self.doubleloop = False
    def append(self,new): #Append edgeNode to edgeList
        if self.start:
            oldend = self.end
            oldend.child = new           
            self.end = new
            self.end.parent = oldend
            self.endnode = self.end.opposite(self.endnode)
        else:
            self.start = new
            self.parent = None
            self.end = new
            self.endnode = self.start.opposite(self.startnode)
    def total_distance(self):
        self.length = 0
        for edge in openPath(self):
            self.length += edge.distance()
    def close_loop(self):
        self.end.child = self.start
        self.start.parent = self.end
    def cut_cycle(self,start):
        self.unmark_nodes(start,self.endnode)
        startparent = start.parent
        newCycle = edgeList(self.endnode,self.endnode,start,self.end) #Since startnode and endnode are identical that should tell the constructor that it is initializing a loop.
        if startparent != None:
            self.end = startparent #Cut the new cycle off from the AB_path
            self.endnode = self.endnode #Just to emphasize that when cutting a loop off of a path the endnode will remain the same
            self.remark_path()
        else:
            self.start = None #If the new cycle was all of the edges in the loop then the loop should be empty save for its startnode.
            self.end = None
            self.endnode = None #Can never be too safe.
        return newCycle
    """
        Traverse each node in the path backwards until we find the edge starting the loop from the endnode.
    """
    def back_up(self,endnode): 
        previous_node = endnode
        current_edge = self.end
        if self.startnode == endnode and self.doubleloop: #This condition guarantees that we are closing a double loop.
            """
                The only edge we could be looking for when we are closing a double loop is the start edge. If there has already been a false
                close, then both edges leading out of false close node from one graph have been taken. AND if the algorithm has found the start node 
                of the path after a false close then it did so traversing an edge from B and any starting edge for an AB_path is an A edge
                so closes on a double loop are always valid. Double loops, in this definition, are AB_cycles that contain a false close on the start node of the 
                AB_path that generated the cycle. These exclude AB_cycles that have a false close anywhere else, as when the algorithm false closes on
                any other node, it can never return to that node on the same AB_path much less in the rest of the cycle generating algorithm. Therefore,
                there is no need for a special case for handling these cycles and no need for a boolean variable to denote them. All I could think of
                when I was coming up with the name for the varaible was double loop anyways. 
            """
            return self.start #Thats alot to say just return the start edge.
        while True:
            next_node = current_edge.opposite(previous_node)
            if next_node.id == endnode.id:
                return current_edge
            current_edge = current_edge.parent
            previous_node = next_node
    """
        This operation is required to unmark all edges in the AB_path so re-encountering them again after they have been used in an AB_cycle doesn't make the program 
        assume the node is somewhere on the current AB_path and thus cause back_up to throw an exception. These nodes are simply not part of the AB_path once the
        cycle that contains them has been cut from the AB_path.

        Starts from startnode in the AB_path and unmarks each node in the path until every node after startnode is unmarked. 
        This is done before the edges are cut off and made into a cycle
    """
    def remark_path(self):
        if self.start == None:
            return
        currentnode = self.startnode
        currentedge = self.start
        while True:
            currentnode.on_path = True
            if currentedge == self.end:
                break
            currentnode = currentedge.opposite(currentnode)
            currentedge = currentedge.child     
            
    def unmark_nodes(self,startedge,startnode): #Unmark all of the nodes in the cycle to be cut from self. This operation only unmarks nodes and makes no changes otherwise.
        currentnode = startnode
        currentedge = startedge
        while True: #First mark the node, then check if the edge after it is the last in the cycle. This will cover all the nodes in the cycle as the last node is also the first.
            currentnode.on_path = False
            """
                If the new cycle has a false close node that is also the start node of the AB_path, we can make a path without closing the entire double loop. The
                B edge leading back into this false close can still be found again. Just think about it as the AB_path getting another chance to close its
                start node with out running into a false close the first time.
            """
            if self.doubleloop and currentnode.id == self.startnode.id: #!!!THIS SHOULD ACTUALLY BE THE START OF THE AB_PATH!!!
                self.doubleloop = False
            if currentedge == self.end:
                break
            currentnode = currentedge.opposite(currentnode)
            currentedge = currentedge.child

    """
        Returns the next node in the multigraph Gab to be arrived at. 
        Sets flags to indicate which paths have been traversed and if this node has been visited before.
    """

    def pass_through(self,current):
        if self.end:
            trueifA = not self.end.A # trueifA is true if the next edge to traverse belongs to A.
        else:
            trueifA = True
        child = current.edges[2*int(not trueifA)] #Path leads to the child of cur node
        parent = current.edges[1 + 2*int(not trueifA)]
        if not child.taken and not parent.taken:
            r = random()
            if r <= .5:
                next = Gab[child.child_id]
                newedge = child
            else:
                next = Gab[parent.parent_id]
                newedge = parent
        else: #Figure out when to cross off vertices in permutation
            permutation.mark(current.id) #If the node has been passed through two times it is gauranteed to never be traversed again or it will be closed before the next jump.
            if not child.taken:
                next = Gab[child.child_id]
                newedge = child
            elif not parent.taken:
                next = Gab[parent.parent_id]
                newedge = parent
            else: #If there are no more edges coming from current node that can be traversed, return no node.
                return None
        current.on_path = True
        newedge.taken = True
        self.append(newedge)
        return next
    
def generate_path():
    permutation.shuffle(PROBLEM_SIZE)
    new = path()
    for cur in permutation.deck:
        new.append(pathNode(cur))
    new.close_loop()
    return new

class path:
    def __init__(self):
        self.start = None
        self.end = None
        self.arrayform = None #Random access allows O(N) overlap of paths. Make adhoc because especially fit paths will be crossed over more than poor fitness.
        self.mutatearray = None
        self.length = 0
        self.edgecount = 0
    def __lt__(self,other):
        return self.edgecount < other.edgecount
    def arrayInit(self):
        if self.arrayform:
            return
        else:
            self.arrayform = [None]*PROBLEM_SIZE#Update this to be dynamic later.
            for current in openPath(self):
                self.arrayform[current.id] = current
    def mutate_arrayInit(self):
        self.mutatearray = []
        for node in openPath(self):
            self.mutatearray.append(node)
    def append(self,newend): #Append pathNode to path
        if self.start:
            oldend = self.end
            oldend.addChild(newend)
            length = pathNode.distance(newend,oldend)
            self.length += length
            self.end = newend
            self.edgecount += 1
        else:
            self.start = newend
            self.end = newend
    def close_loop(self):
        self.start.parent = self.end
        self.end.child = self.start
        self.length += self.end.distance(self.start)
        self.edgecount += 1
    def reset_flags(self):
        for node in openPath(self):
            node.reset_flags()
    def resetend(self):
        self.end = self.start.parent
    def recalculatedist(self):
        self.length = 0
        for node in openPath(self):
            self.length += pathNode.distance(node,node.child)

    def reconfig(self,out1,out2,out3,code):
        if code == 2:
            temp = out1.child
            out1.child = out2.child
            out1.child.parent = out1
            out2.child = out3.child
            out2.child.parent = out2
            out3.child = temp
            out3.child.parent = out3
        else:
            raise Exception("Unknown reordering code")
        self.end = self.start.parent
        

class pathNode:
    def __init__(self,id,A = False):
        self.id = id
        self.arrayform = None  
        self.child = None
        self.parent = None
    def addChild(self,child):
        self.child = child
        self.child.parent = self
    def addParent(self,parent):
        self.parent = parent
        self.parent.child = self
    def intersect(self,intersection): #When making the multigraph, (Gab) simply add identical points as an attribute.
        self.intersection = intersection
        self.intersection.intersection = self
    def getx(self):
        return nodearray[self.id].x
    def gety(self):
        return nodearray[self.id].y
    def distance(self,other):
        return point.distance(nodearray[self.id],nodearray[other.id])
    def reset_flags(self):
        self.childarc = False
        self.parentarc = False
        self.A = False
        self.found = False

class linkedlistIterator:
    def __init__(self,list):
        self.start = list.start
        self.current = self.start
    def __iter__(self):
        return self
    def __next__(self):
        if self.current == None:
            raise StopIteration
        temp = self.current
        self.current = self.current.child
        return temp

class closedPath: #Treats path like it is closed aby iterating over the start twice
    def __init__(self,path):
        self.start = path.start
        self.current = self.start
        """
            This flag indicates to the iterator that it has passed all of the nodes in the circuit and returned the start node for the second time.
            If the program tries to call the iterator after this flag is set, __next__ will raise the StopIteration exception.
        """
        self.fullCircuit = False
        """
            This flag being set indicates that the iterator has returned the start node. This way, full circuit wot be set the first time 
            the start node is returned.
        """
        self.firstCall = False
    def __iter__(self):
        return self
    def __next__(self):
        if self.fullCircuit:
            raise StopIteration
        temp = self.current
        self.current = self.current.child
        self.fullCircuit = (temp == self.start) and (self.firstCall) #Indicates that the start node has been traversed again.
        self.firstCall = True#Its probably more efficient to just set firstcall to true over and over again instead of using an if statement.
        return temp

class openPath: #Iterator that treats closed path like open paths. 
    def __init__(self,path):
        self.start = path.start
        self.current = self.start
        """
            This flag indicates to the iterator that it has passed all of the nodes in the circuit and returned the start node for the second time.
            If the program tries to call the iterator after this flag is set, __next__ will raise the StopIteration exception.
        """
        self.fullCircuit = False
        """
            This flag being set indicates that the iterator has returned the start node. This way, full circuit wont be set the first time 
            the start node is returned.
        """
        self.firstCall = False
    def __iter__(self):
        return self
    def __next__(self):
        if self.current == self.start and self.firstCall:
            raise StopIteration
        temp = self.current
        self.current = self.current.child
        self.firstCall = True#Its probably more efficient to just set firstcall to true over and over again instead of using an if statement.
        return temp
    
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

    graphics_commands.cycle_cover(closedPath(A),closedPath(B),cycles,edge_cycle_to_nodes,"Cycle Cover")

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

    graphics_commands.draw_paths(subtours,closedPath,"Subtours")

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

def child_health(child,expected_weight_of_the_newborn_infant = -1):
    if expected_weight_of_the_newborn_infant == -1:
        expected_weight_of_the_newborn_infant = PROBLEM_SIZE
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