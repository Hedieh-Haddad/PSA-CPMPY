# how many 'palindrome days' are there in a century?
# for example, today: 121121 (12 november 2021)
# Thanks for asking Michiel : )

from cpmpy import *
from cpmpy.tools import ParameterTuner


v = intvar(0,9, shape=6)
day = 10*v[0] + v[1]
month = 10*v[2] + v[3]
year = 10*v[4] + v[5]
# for the American version:
#month = 10*v[0] + v[1]
#day = 10*v[2] + v[3]

m = Model(
    day >= 1, day <= 31,
    month >= 1, month <= 12,
    year >= 0, year <= 99,
    v[0] == v[-1],
    v[1] == v[-2],
    v[2] == v[-3],
    (month == 2).implies(day <= 28) # february
)
for no31 in [2,4,6,9,11]:
    m += [(month == no31).implies(day<=30)]

c = 0
while m.solve():
    c += 1
    print(c, v.value())
    m += ~all(v == v.value())

tunables = {
    "search_branching": [0, 1, 2, 3, 4, 5, 6, 7],
    "linearization_level": [0, 1],
    'symmetry_level': [0, 1, 2]}

defaults = {
    "search_branching": 7,
    "linearization_level": 0,
    'symmetry_level': 1}

solver = "ortools"
tuner = ParameterTuner(solver, m, tunables, defaults)

default_params = {
    "init_round_type": "Static",
    "stop_type": "Timeout",
    "tuning_timeout_type": "Static",
    "time_evol": "Dynamic_Geometric",
    "HPO": "Bayesian"
}
user_params = {
    "init_round_type": "Static",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Static",  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
    "HPO": "Bayesian",  # "Hamming", "Bayesian", "Grid"
}

params = {**default_params, **user_params}

best_params = tuner.tune(
    time_limit=40,
    max_tries=10,
    **params
)
best_runtime = tuner.best_runtime

print(best_params)
print(best_runtime)
