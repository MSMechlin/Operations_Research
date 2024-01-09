from queue import PriorityQueue
from Debugging import *
"""
    Classes: Uppercase initial
    Members/Local: variables camelcase
    Action Methods: lowercase underscore spacing
    Methods that return objects: uppercase underscore space

"""
"""
    This iterator is meant to return each finish_task event from a job
    in order. 
"""

"""
    Fitness can be a partial ordering or we can optimize the singular ordering and hope a feasible soultion will result.
"""
class ResourceQueue:
    def __init__(self):
        self.start = None
        self.end = None
        self.open = True
    def enqueue(self,new):
        if self.start == None:
            self.start = new
            self.end = new
        else:
            self.end.resourceSuccessor = new
            new.resourcePredecessor = self.end
            self.end = new 
    def next(self):
        ret = self.start
        if ret == None or not ret.ready():
            return None
        else:
            self.remove(ret)
            return ret
    def next_available(self):
        current = self.start
        while current != None:
            if current.ready():
                self.remove(current)
                return current
            current = current.resourceSuccessor
        return None
    def remove(self,rem):
        if rem == self.start:
            self.start = self.start.resourceSuccessor
            if self.start:
                self.start.resourcePredecessor = None
        elif rem == self.end:
            self.end = self.end.resourcePredecessor
            self.end.resourceSuccessor = None
        else:
            rem.resourceSuccessor.resourcePredecessor = rem.resourcePredecessor
            rem.resourcePredecessor.resourceSuccessor = rem.resourceSuccessor
class Task:
      def __init__(self,duration,resourceRequirement,jobNumber,taskNumber,taskID):
            self.resource = resourceRequirement
            self.duration = duration
            self.jobNumber = jobNumber
            self.taskNumber = taskNumber
            self.taskID = taskID

class Problem:
    def __init__(self,problemfile):
        self.problemfile = problemfile
        file = open(problemfile,"r")
        """
            ALL FILE PARSING HAPPENS HERE!!!!
        """
        lines=iter(file)
        next(lines)
        line = next(lines)
        self.resourceCount = int(line)
        next(lines)
        jobs = []
        jobNumber = 1
        taskID = 0
        for line in lines:
            tabbed = line.split(" ")
            release = int(tabbed[0])
            tasks = []
            taskNumber = 1
            for item in tabbed[1:]:
                commasplit = item.split(",")
                tasks.append(Task(int(commasplit[0]),int(commasplit[1]),jobNumber,taskNumber,taskID))
                taskNumber += 1
                taskID += 1
            jobs.append(Job(tasks,jobNumber,release))
            jobNumber += 1
        self.jobSet = JobSet(jobs)
        self.size = taskID
    def Resource_Queues(self):
        queues = []
        for i in range(self.resourceCount):
            queues.append(ResourceQueue())
        return queues
    def job_to_task_permutation(self,sequence):
        translation = sequence.copy()
        taskIterators = self.jobSet.Task_Iterators()
        for i in range(len(translation)):
            translation[i] = next(taskIterators[sequence[i]-1]).taskID
        return translation
        
    """
        Take permutation of all task ids and return the sequence of job numbers correpsonding to each.
        Copy sequence to prevent overwriting. 
        Make a habit of not overwriting parameters to functions to prevent cofusion the the case that the
        input parameter needs to be used in its orginal form for aftet the function calls.
        If you want to do otherwise specify.

        For each task id find task corresponding and replace with jobid.
        The conversion will implicitly reorder tasks in the same job to be done in the proper order.

    """
    def task_to_job_permutation(self,sequence):
        tasks = self.jobSet.Total_Task_List()
        translation = sequence.copy()
        for i in range(len(sequence)):
            translation[i] = tasks[sequence[i]].jobNumber
        return translation


class JobSet:
    def __init__(self,jobs):
        self.jobs = jobs
    def Task_Iterators(self): #Returns the tasks.
        taskIterators = []
        for job in self.jobs:
            taskIterators.append(iter(job.tasks))
        return taskIterators
    def Total_Task_List(self):
        ret = []
        for job in self.jobs:
            ret += job.tasks
        return ret
    def Job_Events(self): #Returns event nodes of the tasks.
        taskEvents = []
        releaseEvents = InProcessQueue()
        for job in self.jobs:
            releaseEvent = job.Release_Event()
            releaseEvents.enqueue(releaseEvent,0)
            taskEvents.append(job.Task_Events(releaseEvent))
        return releaseEvents, taskEvents

class JobReleaseEvent:
    def __init__(self,time,jobNumber,eventType):
        self.time = time
        self.type = eventType
        self.jobNumber = jobNumber
        self.done = False
        self.jobSuccessor = None
    def __lt__(self,other):
        return self.time < other.time

class TaskFinishEvent:
    def __init__(self,task:Task,eventType):
        self.time = task.duration
        self.task = task
        self.jobNumber = task.jobNumber
        self.taskNumber = task.taskNumber
        self.type = eventType
        self.done = False
        self.jobPredecessor = None
        self.jobSuccessor = None
        self.resourcePredecessor = None
        self.resourceSuccessor = None
    def ready(self):
        return self.jobPredecessor.done
    def __lt__(self,other):
        return self.time < other.time

class InProcessQueue:
    def __init__(self):
        self.queue = PriorityQueue()
    def enqueue(self,event,time):
        event.time += time
        self.queue.put(event)
    def dequeue(self)-> (TaskFinishEvent | JobReleaseEvent):
        if self.queue.empty():
            return None
        else:
            return self.queue.get()
    def empty(self):
        return self.queue.empty()
RELEASE_JOB = 1
FINISH_TASK = 2        
class Job:
    def __init__(self,tasks,jobNumber,releaseTime):
        self.tasks = tasks
        self.jobNumber = jobNumber
        self.releaseTime = releaseTime         
    """
        This function generates a linked list of Events corresponding
        to the tasks of this job. They start with the release event
        and go in order of succession until the end.
        Initializes the jobPredecessor and jobSuccessor members
        of the event nodes.
    """
    def Release_Event(self):
        return JobReleaseEvent(self.releaseTime,self.jobNumber,RELEASE_JOB)
    def Task_Events(self,release:JobReleaseEvent):
        prevEvent = release
        tasks = iter(self.tasks)
        curEvent = TaskFinishEvent(next(tasks),FINISH_TASK)
        startEvent = curEvent
        curEvent.jobPredecessor = prevEvent
        prevEvent.jobSuccessor = curEvent
        prevEvent = curEvent
        for task in tasks:
                curEvent = TaskFinishEvent(task,FINISH_TASK)
                curEvent.jobPredecessor = prevEvent
                prevEvent.jobSuccessor = curEvent
                prevEvent = curEvent
        return iter(EventIterator(startEvent))

class EventIterator:
    def __init__(self,start:TaskFinishEvent):
        self.start = start  
    def __iter__(self):
        self.current = self.start
        return self
    def __next__(self):
        if self.current == None:
            raise StopIteration
        ret = self.current
        self.current = self.current.jobSuccessor
        return ret

def fitness_1(problem:Problem,order,enableGantt=False):
    return _fitness(problem,order,ResourceQueue.next,enableGantt)

def fitness_2(problem:Problem,order,enableGantt=False):
    return _fitness(problem,order,ResourceQueue.next_available,enableGantt)

def _fitness(problem:Problem,order,nextFunction,enableGantt=False):
    if enableGantt:
        blocks = []
    inProcessQueue, taskEvents = problem.jobSet.Job_Events()
    resourceQueues = problem.Resource_Queues()
    for jobCode in order:
        task = next(taskEvents[jobCode-1])
        resource = task.task.resource
        resourceQueues[resource-1].enqueue(task)
    nextEvent = None
    time = 0
    #MAIN LOOP
    while True:
        resourceQueuesChanged = []
        if nextEvent == None:
            if inProcessQueue.empty():
                break
            nextEvent = inProcessQueue.dequeue()
        time = nextEvent.time
        while nextEvent != None and nextEvent.time == time:
            nextEvent.done = True
            if nextEvent.type == FINISH_TASK:
                if enableGantt:
                    blocks.append((time,nextEvent.task.duration,nextEvent.jobNumber,nextEvent.taskNumber,nextEvent.task.resource))
                resourceNumber = nextEvent.task.resource
                resourceQueuesChanged.append(resourceNumber)
                resourceQueues[resourceNumber-1].open = True
            if nextEvent.type == FINISH_TASK or nextEvent.type == RELEASE_JOB:
                if nextEvent.jobSuccessor != None:
                    resourceQueuesChanged.append(nextEvent.jobSuccessor.task.resource)
            nextEvent = inProcessQueue.dequeue()
        for resourceQueueNumber in resourceQueuesChanged:
            queue = resourceQueues[resourceQueueNumber-1]
            if queue.open:
                enqueue = nextFunction(queue)
                if enqueue:
                    queue.open = False
                    inProcessQueue.enqueue(enqueue,time)
    if enableGantt:
        return time, blocks
    return time