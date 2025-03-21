#!/usr/bin/python3
from cpmpy import *
from cpmpy.tools import ParameterTuner
import numpy as np

# toy model taken from :
# https://python-mip.readthedocs.io/en/latest/examples.html#resource-constrained-project-scheduling
# ported to CPMpy by Guillaume Poveda

# Data
# durations = cpm_array([0, 3, 2, 5, 4, 2, 3, 4, 2, 4, 6, 0])
# durations = cpm_array([0, 5, 3, 8, 6, 4, 7, 6, 5, 9, 10, 3, 6, 8, 5, 7, 4, 3, 5, 8, 9, 4, 7, 5, 6, 8, 9, 7, 3, 0])
#
# # resource_needs = cpm_array([[0, 0], [5, 1], [0, 4], [1, 4], [1, 3], [3, 2], [3, 1], [2, 4], [4, 0], [5, 2], [2, 5], [0, 0]])
# #
# # resource_capacities = cpm_array([6, 8])
# resource_needs = np.random.randint(0, 6, size=(30, 3))
# resource_capacities = cpm_array([10, 12, 15])  # More resources
#
# # successors_link = cpm_array([[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 9], [2, 10], [3, 8], [4, 6], [4, 7], [5, 9], [5, 10], [6, 8], [6, 9], [7, 8], [8, 11], [9, 11], [10, 11]])
# successors_link = cpm_array([
#     [0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 6], [3, 7], [4, 8], [5, 9], [5, 10],
#     [6, 11], [7, 12], [8, 13], [9, 14], [10, 15], [11, 16], [12, 17], [13, 18], [14, 19],
#     [15, 20], [16, 21], [17, 22], [18, 23], [19, 24], [20, 25], [21, 26], [22, 27],
#     [23, 28], [24, 29], [28, 29]
# ])
durations = cpm_array([0] + list(np.random.randint(5, 15, size=99)))  # Jobs with random durations between 5 and 15

# Expanded resource needs (100 jobs x 5 resources for higher complexity)
resource_needs = np.random.randint(0, 6, size=(100, 5))  # 5 resources for 100 jobs
resource_capacities = cpm_array([30, 40, 50, 60, 70])  # More resources

# More complex precedence constraints (randomized for 100 jobs)
successors_link = []
for i in range(100):
    for j in range(i + 1, 100):
        if np.random.rand() < 0.2:  # 20% chance to create a dependency between jobs
            successors_link.append([i, j])

successors_link = cpm_array(successors_link)


nb_resource = len(resource_capacities)
nb_jobs = len(durations)
max_duration = sum(durations)  # dummy upper bound, can be improved of course

# Variables
start_time = intvar(0, max_duration, shape=nb_jobs)


model = Model()
# Precedence constraints
for j in range(successors_link.shape[0]):
    model += start_time[successors_link[j, 1]] >= start_time[successors_link[j, 0]]+durations[successors_link[j, 0]]

# Cumulative resource constraint
for r in range(nb_resource):
    model += Cumulative(start=start_time, duration=durations, end=start_time+durations,
                        demand=resource_needs[:, r], capacity=resource_capacities[r])

makespan = max(start_time)
model.minimize(makespan)

model.solve(solver="ortools")
print("Start times:", start_time.value())


def check_solution(start_time_values):
    for j in range(successors_link.shape[0]):
        assert start_time_values[successors_link[j, 1]] >= start_time_values[successors_link[j, 0]]+\
               durations[successors_link[j, 0]]
    for t in range(max(start_time_values)+1):
        active_index = [i for i in range(nb_jobs) if durations[i] > 0 and
                        start_time_values[i] <= t < start_time_values[i]+durations[i]]
        for r in range(nb_resource):
            consumption = sum([resource_needs[i, r] for i in active_index])
            if consumption>resource_capacities[r]:
                print(t, r, consumption, resource_capacities[r])
            assert consumption <= resource_capacities[r]


check_solution(start_time.value())
print("Solution passed all checks.")



tunables = {
    "search_branching": [0, 1, 2, 3, 4, 5, 6, 7],
    "linearization_level": [0, 1],
    'symmetry_level': [0, 1, 2]}

defaults = {
    "search_branching": 7,
    "linearization_level": 0,
    'symmetry_level': 1}

solver = "ortools"
tuner = ParameterTuner(solver, model, tunables, defaults)

default_params = {
    "init_round_type": "Static",
    "stop_type": "Timeout",
    "tuning_timeout_type": "Static",
    "time_evol": "Dynamic_Geometric",
    "HPO": "Bayesian"
}
user_params = {
    "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Static",  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
    "HPO": "Bayesian",  # "Hamming", "Bayesian", "Grid"
}

params = {**default_params, **user_params}

best_params = tuner.tune(
    time_limit=120,
    max_tries=10,
    **params
)
best_runtime = tuner.best_runtime

print(best_params)
print(best_runtime)
