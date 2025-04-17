import time
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
from ..transformations.normalize import toplevel_list
from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.linearize import canonical_comparison
from ..transformations.reification import only_bv_reifies, reify_rewrite
from ..transformations.comparison import only_numexpr_equality
from ..expressions.utils import is_num
from ..expressions.core import Operator, Comparison
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.variables import _NumVarImpl, _IntVarImpl, _BoolVarImpl, NegBoolView, boolvar, intvar
from pycsp3 import satisfy, functions as f
from cpmpy.expressions.utils import is_num
from cpmpy.expressions.variables import NegBoolView
from cpmpy.expressions.core import BoolVal
from pycsp3 import Var, satisfy
from cpmpy.expressions.variables import _BoolVarImpl
from cpmpy.expressions.variables import NegBoolView
from pycsp3 import SAT
from cpmpy.expressions.variables import NegBoolView
from pycsp3 import Var, satisfy
from cpmpy.expressions.core import BoolVal, Comparison, Operator
from cpmpy.expressions.variables import _BoolVarImpl
from ..expressions.globalconstraints import GlobalConstraint, DirectConstraint, AllEqual
from cpmpy.expressions.utils import is_num
from pycsp3 import Var
from cpmpy.expressions.variables import _BoolVarImpl, _IntVarImpl
from cpmpy.expressions.core import Operator
from cpmpy.expressions.utils import get_bounds
from pycsp3.functions import Var, satisfy, Sum, ScalarProduct, AllDifferent, Not, And, Or, _Intension, \
    _wrap_intension_constraints, imply
from cpmpy.expressions.utils import get_bounds
from cpmpy.expressions.utils import is_num
from pycsp3 import Var
from pycsp3 import solve, values
from pycsp3.classes.main.variables import Variable
from cpmpy.expressions.utils import is_boolexpr
from cpmpy.expressions.utils import get_bounds, is_boolexpr
from cpmpy.expressions.utils import get_bounds
from pycsp3 import satisfy, And, Not, Or
from pycsp3.functions import _Intension
from cpmpy.expressions.core import Operator
from pycsp3 import And, Or, Not
from pycsp3 import minimize as pymin, maximize as pymax
from cpmpy.expressions.core import Operator
from pycsp3 import And, Or, Not


class CPM_ace(SolverInterface):
    @staticmethod
    def supported():
        try:
            import pycsp3
            return True
        except ImportError:
            return False

    def __init__(self, cpm_model=None, subsolver=None):
        if not self.supported():
            raise Exception("Install the 'pycsp3' library to use this solver: pip install pycsp3")

        import pycsp3
        from pycsp3 import clear
        clear()

        self.obj = None
        self.minimize_obj = None
        self.helper_var = None
        self._varmap = {}
        self._aux_counter = 0

        super().__init__(name="ace", cpm_model=cpm_model)
        self.cpm_status = SolverStatus(self.name)

    def solver_var(self, cpm_var):
        if cpm_var in self._varmap:
            return self._varmap[cpm_var]

        if is_num(cpm_var):
            val = int(cpm_var)
            unique_name = f"const_{val}_{len(self._varmap)}"
            const_var = Var(val, val, id=unique_name)
            return const_var

        name = str(cpm_var.name if hasattr(cpm_var, 'name') else str(cpm_var)).replace("[", "_").replace("]",
                                                                                                         "").replace(
            ",", "_")

        if isinstance(cpm_var, _BoolVarImpl):
            pvar = Var(0, 1, id=name)
            self._varmap[cpm_var] = pvar
            return pvar

        elif isinstance(cpm_var, _IntVarImpl):
            lb, ub = cpm_var.lb, cpm_var.ub
            pvar = Var(lb, ub, id=name)
            self._varmap[cpm_var] = pvar
            return pvar

        elif isinstance(cpm_var, Operator):
            name = cpm_var.name
            if name == "sum":
                args = [self.solver_var(arg) for arg in cpm_var.args]
                expr = Sum(args)
                self._varmap[cpm_var] = expr
                return expr

            elif name == "wsum":
                weights, vars_ = cpm_var.args
                w_args = [int(w) for w in weights]
                x_args = [self.solver_var(v) for v in vars_]
                expr = ScalarProduct(w_args, x_args)
                self._varmap[cpm_var] = expr
                return expr

            elif name == "mul":
                a, b = [self.solver_var(arg) for arg in cpm_var.args]
                self._aux_counter += 1
                aux_name = f"aux_{self._aux_counter}"
                aux = Var(0, max(100, 10 ** 6), id=aux_name)  # give a broad enough domain
                satisfy(aux == a * b)
                self._varmap[cpm_var] = aux
                return aux
            else:
                lb, ub = get_bounds(cpm_var)
                self._aux_counter += 1
                aux_name = f"aux_{self._aux_counter}"
                aux = Var(lb, ub, id=aux_name)
                self._varmap[cpm_var] = aux

                if is_boolexpr(cpm_var):
                    expr = self._convert_boolexpr(cpm_var)
                    satisfy((aux == 1) == expr)
                else:
                    self._get_constraint(cpm_var == aux)

                return aux

    def _get_constraint(self, cpm_expr):
        # print("cpm_expr", cpm_expr)
        if isinstance(cpm_expr, NegBoolView):
            orig = self.solver_var(cpm_expr._bv)
            satisfy(orig != 1)
            return None

        elif isinstance(cpm_expr, _BoolVarImpl):
            satisfy(self.solver_var(cpm_expr) == 1)
            return None

        elif isinstance(cpm_expr, BoolVal):
            if cpm_expr.args[0] is True:
                return None
            else:
                dummy = Var(0, 0)
                satisfy(dummy < 0)
                return None

        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            op = cpm_expr.name

            lhs_var = self.solver_var(lhs)
            rhs_var = self.solver_var(rhs)

            if op == "==":
                satisfy(lhs_var == rhs_var)
            elif op == "!=":
                satisfy(lhs_var != rhs_var)
            elif op == "<":
                satisfy(lhs_var < rhs_var)
            elif op == "<=":
                satisfy(lhs_var <= rhs_var)
            elif op == ">":
                satisfy(lhs_var > rhs_var)
            elif op == ">=":
                satisfy(lhs_var >= rhs_var)
            else:
                raise NotImplementedError(f"Comparison operator {op} not supported")
            return None

    def __add__(self, cpm_expr):
        from cpmpy.transformations.get_variables import get_variables

        get_variables(cpm_expr, collect=self.user_vars)
        for con in self.transform(cpm_expr):
            self._get_constraint(con)
        return self

    def solve(self, time_limit=None, **kwargs):
        for var in self.user_vars:
            self.solver_var(var)
        start = time.time()
        status = solve()
        end = time.time()
        self.cpm_status.runtime = end - start
        if not status:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            return False
        else:
            self.cpm_status.exitstatus = ExitStatus.OPTIMAL
            for cpm_var in self.user_vars:
                solver_var = self._varmap.get(cpm_var, None)
                if isinstance(solver_var, Variable) and solver_var.values:
                    cpm_var._value = values(solver_var)
            return True

    def transform(self, cpm_expr):
        cpm_cons = toplevel_list(cpm_expr)
        supported = {"alldifferent", "sum", "element"}

        cpm_cons = decompose_in_tree(cpm_cons, supported, supported)
        cpm_cons = flatten_constraint(cpm_cons)
        cpm_cons = canonical_comparison(cpm_cons)
        cpm_cons = reify_rewrite(cpm_cons, supported=supported | {"sum", "wsum"})
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported={"sum", "wsum"})
        return cpm_cons

    def objective(self, expr, minimize=True):
        obj_var = self.solver_var(expr)
        # print("DEBUG objective expr:", expr)
        # print("DEBUG solver_var(expr):", obj_var)
        self.objective_value_ = expr
        self.obj = obj_var
        self.minimize_obj = minimize

        if minimize:
            pymin(obj_var)
        else:
            pymax(obj_var)

    # def objective_value(self):
    #     return self.obj_value

    def has_objective(self):
        return self.obj is not None

    def _convert_boolexpr(self, expr):
        if isinstance(expr, NegBoolView):
            bv = self.solver_var(expr._bv)
            return Not([bv == 1])

        elif isinstance(expr, _BoolVarImpl):
            return self.solver_var(expr) == 1

        elif isinstance(expr, Operator):
            args = [self._convert_boolexpr(arg) for arg in expr.args]
            if expr.name == "and":
                return And(args)
            elif expr.name == "or":
                return Or(args)
            elif expr.name == "->":
                a, b = args
                return Or([Not([a]), b])
            elif expr.name == "~":
                return Not([args[0]])
            else:
                raise NotImplementedError(f"Unsupported boolean op: {expr.name}")

        raise NotImplementedError(f"Unsupported boolean expression: {expr}")

