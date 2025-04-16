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
from pycsp3.functions import Var, satisfy, Sum, ScalarProduct, AllDifferent, Not, And, Or, _Intension, _wrap_intension_constraints
from pycsp3.classes.entities import ECtr, EMetaCtr
from pycsp3.classes.main.constraints import ConstraintIntension
from pycsp3 import *
#
# x = VarArray(size=3, dom={0,1})
# satisfy(
#     Or(x[0] == 1, x[1] == 1, x[2] == 1)
# )
# print("satisfy is done")


class CPM_ace(SolverInterface):
    """
    Interface to ace's API

    Requires that the 'pycsp3' python package is installed:
    $ pip install pycsp3

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - ace_model: object, ace's model object
    """

    @staticmethod
    def supported():
        try:
            import ortools
            return True
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_ace: Install the python package 'pycsp3' to use this solver interface.")

        from pycsp3.solvers.ace import Ace

        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        # [GUIDELINE] we commonly use 3-letter abbrivations to refer to native objects:
        #           OR-tools uses ace_solver, Gurobi grb_solver, Exact xct_solver...
        self.ace_solver = pycsp3.solver("ACE")
        self.ace_model = cpm_model

        # for the objective
        self.obj = None
        self.minimize_obj = None
        self.helper_var = None
        super().__init__(name="ACE", cpm_model=cpm_model)

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on __add__ which uses the above solver native object(s)



    def solve(self, time_limit=None, **kwargs):
        """
            Call the ace solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))
        self.ace_solver = self.ace_model.get_solver()

        start = time.time()
        if time_limit is not None:
            self.ace_solver.set_timelimit_seconds(time_limit)

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        my_status = self.ace_solver.solve(**kwargs)
        # [GUIDELINE] consider saving the status as self.ace_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.ace_solver.time() # wallclock time in (float) seconds

        # translate solver exit status to CPMpy exit status
        if my_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = self.ace_solver.value(sol_var)
                raise NotImplementedError("ace: back-translating the solution values")

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.ace_solver.ObjectiveValue()

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

           Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

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

    # [GUIDELINE] if ace does not support objective functions, you can delete this function definition
    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj)) # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        # [GUIDELINE] if the solver interface does not provide a solver native "numeric expression" object,
        #         _make_numexpr may be removed and an objective can be posted as:
        #           self.ace_solver.MinimizeWeightedSum(obj.args[0], self.solver_vars(obj.args[1]) or similar

        if minimize:
            self.ace_solver.Minimize(obj)
        else:
            self.ace_solver.Maximize(obj)
    def has_objective(self):
        return self.ace_solver.hasObjective()
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
               return self.ace_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.ace_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.ace_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("ace: Not a known supported numexpr {}".format(cpm_expr))
    # `__add__()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported={"alldifferent"})
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        # ...
        return cpm_cons
    def __add__(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for con in self.transform(cpm_expr_orig):
            self._get_constraint(con)
        return self
    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solver_var(self, cpm_var):
        if is_num(cpm_var):
            return cpm_var
        if isinstance(cpm_var, NegBoolView):
            return ~self.solver_var(cpm_var._bv)

        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = boolvar(name=str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = intvar(cpm_var.lb, cpm_var.ub, name=str(cpm_var))
            else:
                raise NotImplementedError(f"Unknown CPMpy var: {cpm_var}")
            self._varmap[cpm_var] = revar
        print(type(self._varmap[cpm_var]))
        return self._varmap[cpm_var]

    def _get_constraint(self, cpm_expr):
        print("cpm_expr", cpm_expr.name)
        if isinstance(cpm_expr, Operator):
            if cpm_expr.name == 'and':
                return self.ace_model.and_(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                bool_exprs = [(self.solver_var(arg) == 1) for arg in cpm_expr.args]
                print("OR args:")
                for b in bool_exprs:
                    print(f"  - {b} ({type(b)})")
                pycsp3_constraints = [self.convert_to_pycsp3_constraint(b) for b in bool_exprs]
                # Use the Or operator with PyCSP3 constraints
                return Or(*pycsp3_constraints)

            elif cpm_expr.name == "->":
                cond, subexpr = cpm_expr.args
                # First case: both are boolean vars
                if isinstance(cond, _BoolVarImpl) and isinstance(subexpr, _BoolVarImpl):
                    neg_cond = ~cond  # This is a NegBoolView
                    subexpr_var = subexpr
                    # Manually convert the negated boolean expression and boolean variable into constraints
                    lhs_var = self.solver_var(neg_cond)
                    rhs_var = self.solver_var(subexpr_var)
                    return ConstraintIntension((lhs_var == 1) | (rhs_var == 1))
                    # Verify that the constraints are of the correct type
                # Second case: cond is a boolean var, subexpr is a constraint
                elif isinstance(cond, _BoolVarImpl):
                    return self._get_constraint(subexpr).implied_by(self.solver_var(cond))
                elif isinstance(subexpr, _BoolVarImpl):
                    return self._get_constraint(cond).implies(self.solver_var(subexpr))
                else:
                    raise ValueError(f"Unexpected implication case: {cpm_expr}")

            elif cpm_expr.name == '~':
                variable = self._convert_to_var_or_expr(cpm_expr.args[0])
                return Not(self.solver_var(variable))  # Ensure that negation is handled properly
            else:
                raise NotImplementedError(f"Unsupported Operator '{cpm_expr.name}' in ACE interface")

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            if isinstance(lhs, _BoolVarImpl):
                lhs = self.solver_var(lhs)
            if isinstance(rhs, _BoolVarImpl):
                rhs = self.solver_var(rhs)
            op = "=" if cpm_expr.name == "==" else cpm_expr.name
            if is_boolexpr(lhs) and is_boolexpr(rhs):
                if isinstance(cpm_expr, _BoolVarImpl):
                    count = len(self._varmap)
                    revar = boolvar(name=f"boolvar_{count}")
                    lhs_var = self.convert_to_pycsp3_var(lhs, suffix="lhs")
                    rhs_var = self.convert_to_pycsp3_var(rhs, suffix="rhs")
                    print(f"lhs_var: {lhs_var.name}, rhs_var: {rhs_var.name}")
                    if op == "==":
                        if is_boolexpr(lhs) and is_boolexpr(rhs):
                            lhs_var = self.convert_to_pycsp3_var(lhs)
                            rhs_var = self.convert_to_pycsp3_var(rhs)
                            return AllEqual([lhs_var, rhs_var])
                        else:
                            lhs_var = self.convert_to_pycsp3_var(lhs)
                            rhs_constr = self._get_constraint(rhs)
                            if hasattr(rhs_constr, "reify_with"):
                                return rhs_constr.reify_with(lhs_var)
                            else:
                                raise TypeError(f"rhs {rhs} does not support reify_with")
                    elif op == "!=":
                        return Not(AllEqual([lhs_var, rhs_var]))
                    else:
                        raise ValueError(f"Unsupported operator for boolexpr: {op}")
                elif isinstance(lhs, _BoolVarImpl):
                    return self.solver_var(lhs)
                elif isinstance(rhs, _BoolVarImpl):
                    return self.solver_var(rhs)
                else:
                    raise ValueError(f"Unexpected reification {cpm_expr}")
            elif isinstance(lhs, _NumVarImpl):
                return self.ace_model.arithm(self.solver_var(lhs), op, self.solver_var(rhs))
            elif isinstance(lhs, Operator) and lhs.name in {'sum', 'wsum', 'sub'}:
                if lhs.name == 'sum':
                    return self.ace_model.sum(self.solver_vars(lhs.args), op, self.solver_var(rhs))
                elif lhs.name == "sub":
                    a, b = self.solver_vars(lhs.args)
                    return self.ace_model.arithm(a, "-", b, op, self.solver_var(rhs))
                elif lhs.name == 'wsum':
                    wgt, x = lhs.args
                    w = np.array(wgt).tolist()
                    x = self.solver_vars(lhs.args[1])
                    return self.ace_model.scalar(x, w, op, self.solver_var(rhs))
                elif op == '==':
                    ace_rhs = self._to_var(rhs)
                    if isinstance(lhs, Operator):
                        if lhs.name in {'min', 'max', 'abs', 'div', 'mod', 'element', 'nvalue'}:
                            ace_args = self._to_vars(lhs.args)
                            if lhs.name == 'min':
                                return self.ace_model.min(ace_rhs, ace_args)
                            elif lhs.name == 'max':
                                return self.ace_model.max(ace_rhs, ace_args)
                            elif lhs.name == 'abs':
                                assert len(ace_args) == 1, f"Expected one argument for abs, got {ace_args}"
                                return self.ace_model.absolute(ace_rhs, ace_args[0])
                            elif lhs.name == "div":
                                dividend, divisor = ace_args
                                return self.ace_model.div(dividend, divisor, ace_rhs)
                            elif lhs.name == 'mod':
                                dividend, divisor = ace_args
                                return self.ace_model.mod(dividend, divisor, ace_rhs)
                            elif lhs.name == "element":
                                arr, idx = ace_args
                                return self.ace_model.element(ace_rhs, arr, idx)
                            elif lhs.name == "nvalue":
                                return self.ace_model.n_values(ace_args, ace_rhs)
                        elif lhs.name == 'count':
                            arr, val = lhs.args
                            return self.ace_model.count(self.solver_var(val), self._to_vars(arr), ace_rhs)
                        elif lhs.name == "among":
                            arr, vals = lhs.args
                            return self.ace_model.among(ace_rhs, self._to_vars(arr), vals)
                        elif lhs.name == 'mul':
                            a, b = self.solver_vars(lhs.args)
                            if isinstance(a, int):
                                a, b = b, a
                            return self.ace_model.times(a, b, ace_rhs)
                        elif lhs.name == 'pow':
                            return self.ace_model.pow(*self.solver_vars(lhs.args), ace_rhs)
                        else:
                            raise NotImplementedError(
                                f"Unsupported operator on LHS of equality: '{lhs.name}' in {cpm_expr}")
        elif isinstance(cpm_expr, GlobalConstraint):
            if cpm_expr.name in {"alldifferent", "alldifferent_except0", "allequal", "circuit", "inverse",
                                 "increasing", "decreasing", "strictly_increasing", "strictly_decreasing",
                                 "lex_lesseq", "lex_less"}:
                ace_args = self._to_vars(cpm_expr.args)
                if cpm_expr.name == 'alldifferent':
                    return self.ace_model.all_different(ace_args)
                elif cpm_expr.name == 'alldifferent_except0':
                    return self.ace_model.all_different_except_0(ace_args)
                elif cpm_expr.name == 'allequal':
                    return self.ace_model.all_equal(ace_args)
            elif cpm_expr.name == 'InDomain':
                assert len(cpm_expr.args) == 2
                expr, table = self.solver_vars(cpm_expr.args)
                return self.ace_model.member(expr, table)
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = cpm_expr.args
                start, end, demand, cap = self._to_vars([start, end, demand, cap])
                dur = self.solver_vars(dur)
                tasks = [self.ace_model.task(s, d, e) for s, d, e in zip(start, dur, end)]
                return self.ace_model.cumulative(tasks, demand, cap)
            elif cpm_expr.name == "gcc":
                vars, vals, occ = cpm_expr.args
                return self.ace_model.global_cardinality(*self.solver_vars([vars, vals]), self._to_vars(occ),
                                                         cpm_expr.closed)
            else:
                raise NotImplementedError(
                    f"Unknown global constraint {cpm_expr}, should be decomposed! If you reach this, please report on github.")
        elif isinstance(cpm_expr, _BoolVarImpl):
            return self.solver_var(cpm_expr)

        elif isinstance(cpm_expr, BoolVal):
            if cpm_expr.args[0] is True:
                return None
            else:
                if self.helper_var is None:
                    self.helper_var = self.ace_model.intvar(0, 0)
                return self.ace_model.arithm(self.helper_var, "<", 0)
        elif isinstance(cpm_expr, DirectConstraint):
            c = cpm_expr.callSolver(self, self.ace_model)
            return c
        raise NotImplementedError(cpm_expr)

    def _convert_to_variable(self, var):
        # Convert _BoolVarImpl to Variable
        if isinstance(var, _BoolVarImpl):
            return boolvar(name=str(var))
        return var

    def convert_to_pycsp3_var(self, cpm_var, suffix=""):
        """Convert CPMpy variable to pycsp3 variable with a unique name."""
        if isinstance(cpm_var, _BoolVarImpl):
            # Generate a unique name by appending a suffix
            unique_name = f"{cpm_var.name}_{suffix}"
            pycsp3_var = Var(unique_name)  # Ensure this is a valid conversion
            if pycsp3_var is None:
                raise ValueError(f"Failed to convert {cpm_var} to a pycsp3 variable.")
            return pycsp3_var
        else:
            raise TypeError(f"Unsupported type for conversion: {type(cpm_var)}")

    def bool_to_constraint(self, expr):
        if isinstance(expr, _BoolVarImpl):
            return self.solver_var(expr) == 1
        elif isinstance(expr, Expression) and expr.name == "~":
            return self.solver_var(expr.args[0]) == 0
        else:
            return expr  # Already a valid PyCSP3 constraint

    def _to_constraint(self, var):
        if isinstance(var, _BoolVarImpl):
            return var == 1
        return var  # Already a constraint

    def convert_to_pycsp3_constraint(self, bool_var):
        # Assuming that bool_var is a CPMpy _BoolVarImpl object
        # Convert the CPMpy boolean variable to a PyCSP3 constraint (ECtr)
        # In this case, we assume the boolean variable can be converted to an ECtr in PyCSP3
        return ECtr(bool_var)  # Adjust this conversion if needed
