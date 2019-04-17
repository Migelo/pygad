'''
A module for controlled evaluation of Python expression.

By default the evaluator does not allow to access any underscore identifiers that
can potentially be hazardous to the system (compare e.g.
'eval("__import__('os').system('rm -rf /*')", {})' -- DO NOT TRY THIS!). In this
way the semi-private underscore attributes are actually protected at the same
time.

The evaluation of expression is also better controllable and debuggable with this
module as one can follow the evaluation process.

The drawback is, that the evaluation is slightly slower than the one of the
built-in 'eval' (by a few microseconds per call on my machine and a bit more for
the convenience 'eval').

Examples:
    >>> eval('7+4')
    11
    >>> eval('A/B*log10(12**3)/pi', {'B':1.2}, A=1.234)
    1.0597408738693963
    >>> eval('__import__("os")')
    Traceback (most recent call last):
    ...
    pygad.utils.safe_eval.EvalError: "Undefined symbol: '__import__'"
    >>> eval('arange(10).__class__')
    Traceback (most recent call last):
    ...
    pygad.utils.safe_eval.EvalError: "accessing underscore attributes is not allowed!: 'arange(10).__class__'"
    >>> e = Evaluator({'c':42}, my_math=np)
    >>> e.eval('pi*c')
    131.94689145077132
    >>> e.eval('exp(arr**2)', {'arr':np.arange(6)})
    array([1.00000000e+00, 2.71828183e+00, 5.45981500e+01, 8.10308393e+03,
           8.88611052e+06, 7.20048993e+10])
    >>> for name in iter_idents_in_expr('test.attr + more'):
    ...     print(name)
    attr
    more
    test
    >>> e.eval('a%2 == 0', {'a':np.arange(6)})
    array([ True, False,  True, False,  True, False])
    >>> e.eval('(a%2 == 0) | ((a%3 == 0))', {'a':np.arange(6)})
    array([ True, False,  True,  True,  True, False])
'''
__all__ = ['iter_idents_in_expr', 'EvalError', 'Evaluator', 'eval']

import math
import numpy as np
import operator as op
import ast
import re
import numbers
from fractions import Fraction

def iter_idents_in_expr(expr, retel=False):
    '''
    Iterate all identifiers within the Python expression.

    Args:
        expr (str):     A Python expression.
        retel (bool):   Also yield the AST element itself.

    Yields:
        ident (str):    The identifier.
        el (ast-type):  The AST element (ast.Name or ast.Attribute).
    '''
    for el in ast.walk(ast.parse(expr)):
        name = None
        if isinstance(el, ast.Name):
            name = el.id
        elif isinstance(el, ast.Attribute):
            name = el.attr
        if name:
            if retel:
                yield name, el
            else:
                yield name

class EvalError(Exception):
    '''
    Exception class for any errors that arise from the evaluator class.

    Args:
        msg (object):   The message to be displayed.
    '''
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class Evaluator(object):
    '''
    A class to evaluation expressions in a controlled way.

    See module documentation for more information.

    Args:
        namespace (dict):   Additional constants (numbers only) and callables to
                            use.
        my_math (module):   The module to take the standard functions and
                            constants from. The standard choices are math and
                            numpy. (default: numpy)
        truediv (bool):     Whether to use true division (1/2 -> 0.5).
        allow_underscore (bool):
                            Whether to allow to access underscore names.
    '''

    def __init__(self, namespace=None, my_math=None, truediv=True,
                 allow_underscore=False):
        if my_math is None:
            my_math = np

        self.allow_underscore = allow_underscore

        self.bin_op = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv if truediv else op.div,
                ast.Mod: op.mod,
                ast.Pow: op.pow,
                ast.Eq: op.eq,
                ast.Gt: op.gt,
                ast.GtE: op.ge,
                ast.Lt: op.lt,
                ast.LtE: op.le,
                ast.Is: op.is_,
                ast.IsNot: op.is_not,
                ast.BitAnd: np.bitwise_and,
                ast.BitOr: np.bitwise_or,
                ast.BitXor: np.bitwise_xor,
                }
        self.un_op = {
                ast.UAdd: op.pos,
                ast.USub: op.neg,
                ast.Not: op.not_,
                }

        self.namespace = {'None':None, 'True':True, 'False':False,
                'Fraction':Fraction}
        for ns in [my_math.__dict__, namespace if namespace is not None else {}]:
            self.namespace.update( { name:val for name,val in ns.items()
                    if hasattr(val,'__call__')
                    or isinstance(val,(numbers.Number,str,np.ndarray)) } )

    def _eval(self, node):
        if hasattr(node,'ctx') and not isinstance(node.ctx, ast.Load):
            raise EvalError('only loading context allowed')
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Name):
            return self.variables[node.id]
        elif isinstance(node, ast.BinOp):
            return self.bin_op[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.un_op[type(node.op)](self._eval(node.operand))
        elif isinstance(node, ast.Call):
#           the following lines are removed: incompatible since Pyhton 3.5
#           if node.kwargs:
#               raise NotImplementedError('kwargs are not supported')
#           if node.starargs:
#               raise NotImplementedError('starargs are not supported')
            kwargs = { kw.arg: self._eval(kw.value) for kw in node.keywords }
            posargs = [self._eval(arg) for arg in node.args]
            return self._eval(node.func)(*posargs,**kwargs)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('_') and not self.allow_underscore:
                raise EvalError('accessing underscore attributes is not allowed!')
            return getattr(self._eval(node.value), node.attr)
        elif isinstance(node, ast.IfExp):
            if self._eval(node.test):
                return self._eval(node.body)
            else:
                return self._eval(node.orelse)
        elif isinstance(node, ast.Compare):
            left = self._eval(node.left)
            if len(node.ops) == 1:
                # might be an array-like comparison, that does not work in the
                # general case of multiple comparisons at once
                operator, comparator = node.ops[0], node.comparators[0]
                right = self._eval(comparator)
                return self.bin_op[type(operator)](left, right)
            for operator,comparator in zip(node.ops,node.comparators):
                right = self._eval(comparator)
                if not self.bin_op[type(operator)](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                val = self._eval(val)
                if isinstance(node.op, ast.Or) and val:
                    return True
                if isinstance(node.op, ast.And) and not val:
                    return False
            return isinstance(node.op, ast.And)
        elif isinstance(node, ast.Subscript):
            return self._eval(node.value)[self._eval(node.slice)]
        elif isinstance(node, ast.Slice):
            start = self._eval(node.lower)
            stop  = self._eval(node.upper)
            step  = self._eval(node.step)
            return slice(start, stop, step)
        elif isinstance(node, ast.ExtSlice):
            return tuple(map(self._eval, node.dims))
        elif isinstance(node, ast.Index):
            return self._eval(node.value)
        elif isinstance(node, ast.Str):
            return node.s
        elif node is None:
            return None
        elif isinstance(node, ast.List):
            return list(map(self._eval, node.elts))
        elif isinstance(node, ast.Tuple):
            return tuple(map(self._eval, node.elts))
        elif isinstance(node, ast.NameConstant):
            return node.value
        else:
            # seldomly reached due to substitution of spaces
            raise EvalError('AST node type %s not supported' %
                                node.__class__.__name__)

    def eval(self, expr, idents=None):
        '''
        Evaluate the expression.

        Args:
            expr (str):     The expression with Python syntax to evaluate.
            idents (dict):  Additional identifiers to use during the evalutations.

        Returns:
            value (object): The value of the expression.

        Raises:
            EvalError:      If some used Python syntax is not supported by this
                            Evaluator or of there was an unknown value in the
                            expression.
            others:         Might occur during the evaluation, e.g. OverflowError.
        '''
        self.variables = self.namespace.copy()
        if idents is not None:
            self.variables.update(idents)
        try:
            return self._eval(ast.parse(expr,mode='eval').body)
        except KeyError as e:
            if re.match(r'<class .*>', str(e)):
                raise EvalError('Undefined operation: %s' % e)
            else:
                raise EvalError('Undefined symbol: %s' % e)
        except (EvalError, NotImplementedError) as e:
            raise EvalError("%s: '%s'" % (str(e).strip("'"), expr))
        except:
            raise

def eval(expr, namespace=None, my_math=None, truediv=True,
         allow_underscore=False, **kwargs):
    '''
    Evaluate an expression.

    This is a convenience function to call Evaluator().eval. Hence, for more
    details see Evaluator and Evaluator.eval.

    Args:
        expr (str):         The expression to evaluate.
        namespace (dict):   The namespace to operate on.
        my_math (module):   The module to take the standard functions and
                            constants from. The standard choices are math and
                            numpy. (default: numpy)
        truediv (bool):     Whether to use true division (1/2 -> 0.5).
        allow_underscore (bool):
                            Whether to allow to access underscore names.
        **kwargs:           A convenient way to set varibales in the evaluation
                            namespace.

    Returns:
        value (object):     The evaluated expression.

    Raises:
        EvalError:      If some used Python syntax is not supported by this
                        Evaluator or of there was an unknown value in the
                        expression.
        others:         Might occur during the evaluation, e.g. OverflowError.

    Example:
        >>> eval('2*pi*R', R=2.0)
        12.566370614359172
    '''
    e = Evaluator(namespace=namespace, my_math=my_math, truediv=truediv,
                  allow_underscore=allow_underscore)
    return e.eval(expr, idents=kwargs)
