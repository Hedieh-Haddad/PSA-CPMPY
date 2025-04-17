from datetime import time

import numpy as np
import pycsp3
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, boolvar, intvar
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint, AllEqual
from ..expressions.utils import is_num, is_any_list, eval_comparison, flatlist, argval, argvals, is_boolexpr
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.normalize import toplevel_list
from ..transformations.reification import only_implies, reify_rewrite, only_bv_reifies
from ..transformations.comparison import only_numexpr_equality
from cpmpy import boolvar
from pycsp3.functions import Var, satisfy, Sum, ScalarProduct, AllDifferent, Not, And, Or, _Intension, _wrap_intension_constraints, imply
from pycsp3.classes.entities import ECtr, EMetaCtr
from pycsp3.classes.main.constraints import ConstraintIntension
from pycsp3 import *
from pycsp3.classes.entities import (
    EVar, EVarArray, ECtr, EMetaCtr, ECtrs, EToGather, EToSatisfy, EBlock, ESlide, EAnd, EOr, ENot, EXor, EIfThen, EIfThenElse, EIff, EObjective, EAnnotation,
    AnnEntities, CtrEntities, ObjEntities)
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..expressions.core import Expression, Comparison, Operator
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.utils import is_num, is_any_list, is_boolexpr
from ..transformations.get_variables import get_variables
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.to_cnf import to_cnf
from pycsp3 import clear
from pycsp3 import Var
from pycsp3 import satisfy
from ..expressions.core import Expression, Comparison, Operator, BoolVal
# from cpmpy.expressions.utils import Operator as op


class CPM_template(SolverInterface):
    @staticmethod
    def supported():
        try:
            import pycsp3
            return True
        except ImportError:
            return False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        clear()  # Reset previous model
        self.name = "ace"
        self.varmap = {}  # Mapping from cpmpy Var to pycsp3 Variable

    def solver_var(self, cpm_var, **kwargs):
        dom = cpm_var.domain
        if dom.is_bool():
            pvar = Var(0, 1, id=cpm_var.name)
        else:
            lb, ub = dom.lb(), dom.ub()
            pvar = Var(lb, ub, id=cpm_var.name)
        self.varmap[cpm_var] = pvar
        return pvar

    def transform(self, expr):
        return to_cnf(expr)  # or other transformations if needed

    def __add__(self, expr):
        expr = self.transform(expr)
        if expr.name == "==":
            a, b = expr.args
            satisfy(a == b)
        elif expr.name == "!=":
            a, b = expr.args
            satisfy(a != b)
        elif expr.name == "AllDifferent":
            satisfy(functions.AllDifferent(expr.args))
        else:
            raise NotImplementedError(f"Constraint not supported: {expr}")

    def solve(self, time_limit=None, **kwargs):
        from pycsp3 import solve, values

        status = solve()
        self.objective_value_ = None  # set if optimization
        if not status:
            return False

        for cpm_var, ace_var in self.varmap.items():
            cpm_var._value = values(ace_var)

        return True

    # [GUIDELINE] if TEMPLATE does not support objective functions, you can delete this function definition
    def objective(self, expr, minimize=True):
        from pycsp3 import minimize, maximize

        if minimize:
            minimize(expr)
        else:
            maximize(expr)

    def has_objective(self):
        return self.TPL_solver.hasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Converts a numeric CPMpy 'flat' expression into a solver-specific numeric expression

            Primarily used for setting objective functions, and optionally in constraint posting
        """

        # [GUIDELINE] not all solver interfaces have a native "numerical expression" object.
        #       in that case, this function may be removed and a case-by-case analysis of the numerical expression
        #           used in the constraint at hand is required in __add__
        #       For an example of such solver interface, check out solvers/choco.py or solvers/exact.py

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # any solver-native numerical expression
        if isinstance(cpm_expr, Operator):
           if cpm_expr.name == 'sum':
               return self.TPL_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.TPL_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.TPL_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("TEMPLATE: Not a known supported numexpr {}".format(cpm_expr))


    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):

        # check if objective function
        if self.has_objective():
            raise NotSupportedError("ace does not support finding all optimal solutions")

        # A. Example code if solver supports callbacks
        if is_any_list(display):
            callback = lambda : print([var.value() for var in display])
        else:
            callback = display

        self.solve(time_limit, callback=callback, enumerate_all_solutions=True, **kwargs)
        # clear user vars if no solution found
        if self.ace_solver.SolutionCount() == 0:
            for var in self.user_vars:
                var.clear()
            return self.ace_solver.SolutionCount()

        # B. Example code if solver does not support callbacks
        self.solve(time_limit, enumerate_all_solutions=True, **kwargs)
        solution_count = 0
        for solution in self.ace_solver.GetAllSolutions():
            solution_count += 1
            # Translate solution to variables
            for cpm_var in self.user_vars:
                cpm_var._value = solution.value(self.solver_var(cpm_var))

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display  # callback

        # clear user vars if no solution found
        if solution_count == 0:
            for var in self.user_vars:
                var.clear()

        return solution_count
