"""
    Example of enumerating all Pareto-optimal solutions
    (also called Pareto-efficient or non-dominated solutions)

    Can be seen as an instance of solution-dominance in CSPs
    https://modref.github.io/papers/ModRef2018_SolutionDominance.pdf
"""
from cpmpy.tools import ParameterTuner

def pareto_enumeration(m, objs, solution_limit=None, verbose=False):
    """
        Count (enumerate) all pareto optimal/non-dominated solutions

        Warning! Will modify `m` inplace, take a copy before if unwanted

        It requires repeated solving as well as a filtering pass. The repeated solving can require many solver calls depending on the (arbitrary) choices the solver makes.

        Arguments:
            - m: a Model (or a SolverInterface instance, recommended!)
            - objs: numpy array of functions (CPMpy expressions) to be MAXIMIZED
            - solution_limit: stop after this many solutions (default: None)
            - verbose: if (candidate) solutions should be printed as found

        Returns: list of Pareto optimal solutions found
    """
    # non-dominated (maximization)

    # Forward pass (not dominated by any found before in the sequence):
    seq = []
    while m.solve():
        if verbose:
            print("Pareto candidate solution:", objs.value())
        seq.append(objs.value())
        m += cp.any(objs > objs.value())  # dominance nogood

    # Backward pass (not dominated by any found after in the sequence)
    for i in range(len(seq)-1,-1, -1):  # reverse order
        if any( [not any(seq[i] > seq[j]) for j in range(i+1, len(seq))] ):
            del seq[i]
        elif verbose:
            print("Pareto confirmed non-dominated:", seq[i])

    return seq


if __name__ == "__main__":
    # a toy problem, use its variables as objective function
    import cpmpy as cp
    iv = cp.intvar(1,4, shape=3)
    m = cp.Model(cp.sum(iv) < 10)

    print("Total nr of solutions:", m.solveAll())

    sols = pareto_enumeration(m, iv, verbose=True)
    print("Total nr of Pareto optimal solutions:", len(sols))


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
