from functools import reduce
from json import dump,load

class MarkList:
    def __init__(self,list):
        self.list = list
        self.map = [0] * len(list)
        for i in range(len(list)):
            self.map[list[i]] = i
        self.current = 0
    def mark(self,marked):
        pos = self.map[marked]
        cur = self.current
        if pos < cur:
            return False
        self.current += 1
        self.swap(cur,pos)
        return True
    def swap(self,a,b):
        self.map[self.list[a]] = b
        self.map[self.list[b]] = a
        temp = self.list[a]
        self.list[a] = self.list[b]
        self.list[b] = temp
        return True
    def iscomplete(self):
        return self.current == len(self.list)

class CrossoverMonitor:
    def __init__(self,parent1jobsequence=None,parent2jobseqeunce=None,parent1tasksequence=None,parent2taskseqeunce=None,child1tasksequence=None,child2tasksequence=None,child1jobsequence=None,child2jobsequence=None,*,dictionary=None):
        if dictionary != None:
            d = dictionary
            self = self.__init__(d["p1j"],d["p2j"],d["p1t"],d["p2t"],d["c1j"],d["c2j"],d["c1t"],d["c2t"])
        else:
            self.parent1jobsequence = parent1jobsequence
            self.parent1tasksequence = parent1tasksequence
            self.parent2jobsequence = parent2jobseqeunce
            self.parent2tasksequence = parent2taskseqeunce
            self.child1jobsequence = child1jobsequence
            self.child1tasksequence = child1tasksequence
            self.child2jobsequence = child2jobsequence
            self.child2tasksequence = child2tasksequence
    def __dict__(self):
        dictionary = {}
        dictionary["p1j"] = self.parent1jobsequence
        dictionary["p2j"] = self.parent2jobsequence
        dictionary["p1t"] = self.parent1tasksequence
        dictionary["p2t"] = self.parent2tasksequence
        dictionary["c1j"] = self.child1jobsequence
        dictionary["c2j"] = self.child2jobsequence
        dictionary["c1t"] = self.child1tasksequence
        dictionary["c2t"] = self.child2tasksequence

class CrossoverInput:
    def __init__(self,problemfile=None,parent1=None,parent2=None,i=None,j=None,*,dictionary=None):
        if dictionary:
            d = dictionary
            self = self.__init__(d["problemfile"],d["parent1"],d["parent2"],d["i"],d["j"])
        else:
            self.problemfile = problemfile
            self.parent1 = parent1
            self.parent2 = parent2
            self.i = i
            self.j = j
    def to_dict(self):
        dictionary = {}
        dictionary["problemfile"] = self.problemfile
        dictionary["parent1"] = self.parent1
        dictionary["parent2"] = self.parent2
        dictionary["i"] = self.i
        dictionary["j"] = self.j
        return dictionary

def Task_Count_Struct(problem):
    jobNumber = 0
    taskcountmap = []
    for job in problem.jobSet.jobs:
        count = 0
        for task in job.tasks:
            count += 1
        taskcountmap.append(count)
    return taskcountmap

def Tasks_In_Jobs_Assert(countStruct,child):
    childCountDown = countStruct.copy()
    for task in child.sequence:
        childCountDown[task-1] -= 1
    ret = not any(count != 0 for count in childCountDown)
    return ret

def Task_Complete_Assert(expectedLength,child):
    list=[*range(expectedLength)]
    markList = MarkList(list)
    for task in child:
        repeat = not markList.mark(task)
        if repeat:
            raise Exception("Task Repeated")
    return markList.iscomplete()


def Write_Assert_Object(obj,filename):
    dump(obj.to_dict(),open(filename,"w"))

def Load_Assert_Object(classname,filename):
    dict = load(open(filename,"r"))
    return classname(dictionary = dict)
