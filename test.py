import numpy as np
from cpmpy import *
from cpmpy.tools import ParameterTuner

puzzle_start = np.array([
    [0,3,6],
    [2,4,8],
    [1,7,5]]) # 13 steps

puzzle_end = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,0]])

def n_puzzle(puzzle_start, puzzle_end, N):
    # max nr steps to the solution
    print("Max steps:", N)
    m = Model()

    (dim,dim2) = puzzle_start.shape
    assert (dim == dim2), "puzzle needs square shape"
    n = dim*dim2 - 1 # e.g. an 8-puzzle

    # State of puzzle at every step
    x = intvar(0,n, shape=(N,dim,dim), name="x")

    # Start state constraint
    m += (x[0] == puzzle_start)

    # End state constraint
    m += (x[-1] == puzzle_end)

    # define neighbors = allowed moves for the '0'
    def neigh(i,j):
        # same, left,right, down,up, if within bounds
        for (rr, cc) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
            if 0 <= i+rr and i+rr < dim and 0 <= j+cc and j+cc < dim:
                yield (i+rr,j+cc)

    # Transition: define next based on prev + invariants
    def transition(m, prev_x, next_x):
        # for each position, determine its reachability
        for i in range(dim):
            for j in range(dim):
                m += (next_x[i,j] == 0).implies(any(prev_x[r,c] == 0 for r,c in neigh(i,j)))

        # Invariant: in each step, all cells are different
        m += AllDifferent(next_x)

        # Invariant: only the '0' position can move
        m += ((prev_x == 0) | (next_x == 0) | (prev_x == next_x))

    # apply transitions (0,1) (1,2) (2,3) ...
    for i in range(1, N):
        transition(m, x[i-1], x[i])

    return (m,x)

N = 20 # max nr steps
(m,x) = n_puzzle(puzzle_start, puzzle_end, N)
# Lets minimize the number of steps used...
is_sol = [all((x[i] == puzzle_end).flat) for i in range(N)]
# which means, maximize nr of late steps that are full sol
m.maximize( sum(i*is_sol[i] for i in range(N)) )

if m.solve() is False:
    print("UNSAT, try increasing nr of steps? or wrong input...")
else:
    for i in range(N):
        print("Step", i+1)
        print(x[i].value())
        if (x[i].value() != puzzle_end).sum() == 0:
            # all puzzle_end
            break
print(m.status())

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
