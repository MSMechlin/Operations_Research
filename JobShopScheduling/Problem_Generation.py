import random
import math

"""
The sum of task durations has to be less than the due time.

"""

def generate_problem(filename,resource_count,job_count, max_task_count, min_task_count, max_task_duration, min_task_duration, max_release_time):
    file = open(filename,"w")

    ##NUMBER OF RESOURCES
    file.write("RESOURCES:\n")
    file.write(str(resource_count) + "\n")


    file.write("JOBS:\n")
    for i in range(job_count):
        numberOfTasks = math.floor(random.uniform(min_task_count,max_task_count+1))
        releaseTime = math.floor(random.uniform(0,max_release_time+1))
        file.write("{}".format(str(releaseTime)))
        dueDate = 0
        for i in range(numberOfTasks):
            taskDuration = math.floor(random.uniform(min_task_duration,max_task_duration+1))
            resourceRequirement = math.floor(random.uniform(0,resource_count+1))
            file.write(" {},{}".format(taskDuration,resourceRequirement))
        file.write("\n")


def generate_classical_problem(filename,resource_count,job_count, max_task_count, min_task_count, max_task_duration, min_task_duration, max_release_time):
    file = open(filename,"w")

    ##NUMBER OF RESOURCES
    file.write("RESOURCES:\n")
    file.write(str(resource_count) + "\n")


    file.write("JOBS:\n")
    for i in range(job_count):
        numberOfTasks = math.floor(random.uniform(min_task_count,max_task_count+1))
        releaseTime = math.floor(random.uniform(0,max_release_time+1))
        file.write("{}".format(str(releaseTime)))
        dueDate = 0
        for i in range(numberOfTasks):
            taskDuration = math.floor(random.uniform(min_task_duration,max_task_duration+1))
            resourceRequirement = math.floor(random.uniform(0,resource_count+1))
            file.write(" {},{}".format(taskDuration,resourceRequirement))
        file.write("\n")


import os
filepath = os.path.join(os.path.dirname(__file__), "Problem3.jsh")
generate_problem(filepath, 10, 15, 12, 3, 5, 2, 7)