import time
from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist
import numpy as np
from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus

class PSA:
    @staticmethod
    def geometric_sequence(i):
        return 1.2 * i

    @staticmethod
    def probe(self, solver, opt):
        best_config = self.best_config
        if not self.default_flag:
            params = opt.ask()
            parameters = point_asdict(self.all_params, params)
            self.results_list.append({
                'params': dict(parameters)})
            # print("self.results_list",self.results_list)
            print("Hyper-parameters", self.results_list[-2]['params'])
            if self.results_list[-2]['status'] == ExitStatus.OPTIMAL:
            # if solver.status().exitstatus == ExitStatus.OPTIMAL:
                self.first_non_none_objective = True
                if self.mode == "minimize":
                    if self.results_list[-2]['objective'] < self.best or (self.results_list[-2]['objective'] == self.best and (self.best_runtime is None or self.results_list[-2]['runtime'] < self.best_runtime)):
                    # if obj < best or (obj == best and (self.best_runtime is None or self.actual_runtime < self.best_runtime)):
                        self.best = self.results_list[-2]['objective']
                        best_config = self.results_list[-2]['params']
                        self.best_runtime = self.results_list[-2]['runtime']

                else:
                    # print("YEEEEEEAAAAAAYYYYYYYYYYYYYYYYYYYYYY")
                    # print("best",self.best)
                    if self.results_list[-2]['objective'] > self.best or (self.results_list[-2]['objective'] == self.best and (
                            self.best_runtime is None or self.results_list[-2]['runtime'] < self.best_runtime)):
                    # if obj > best or (obj == best and (self.best_runtime is None or self.actual_runtime < self.best_runtime)):
                        self.best = self.results_list[-2]['objective']
                        best_config = self.results_list[-2]['params']
                        # print("self.actual_runtime",self.actual_runtime)
                        # print("self.best_runtime",self.best_runtime)
                        self.best_runtime = self.results_list[-2]['runtime']
                # print("Best Hyper-parameters before this run", self.best_params)
                print("Objective Value", self.results_list[-2]['objective'])
                # print("Best Objective", self.best)
                # print("********best_config", best_config)
            obj = -self.results_list[-2]['objective'] if self.mode == "maximize" else self.results_list[-2]['objective']
            opt.tell(params, obj)
        best_params = best_config
        return best_params