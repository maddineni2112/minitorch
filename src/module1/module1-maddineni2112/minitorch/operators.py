"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """Returns the product of `x` and `y`."""
    return x * y


def id(x: float) -> float:
    """Returns the input value `x` unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Returns the sum of `x` and `y`."""
    return x + y


def neg(x: float) -> float:
    """Returns the negation of `x`."""
    return -x


def lt(x: float, y: float) -> float:
    """Returns `1.0` if `x` is less than `y`, else `0.0`."""
    return float(x < y)


def eq(x: float, y: float) -> float:
    """Returns `1.0` if `x` is equal to `y`, else `0.0`."""
    return float(x == y)


def max(x: float, y: float) -> float:
    """Returns the maximum of `x` and `y`."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns `1.0` if `x` and `y` are close within `1e-2`, else `0.0`."""
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function.

    If `x >= 0`: `1 / (1 + exp(-x))`
    If `x < 0`:  `exp(x) / (1 + exp(x))`
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU function: returns `x` if `x > 0`, else `0.0`."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Computes the natural logarithm of `x`."""
    return math.log(x)


def exp(x: float) -> float:
    """Computes `e^x`."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of `log(x)` multiplied by `d`: `d / x`."""
    return d / x


def inv(x: float) -> float:
    """Computes the reciprocal of `x`: `1/x`."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of `1/x` multiplied by `d`: `-d / (x^2)`."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU: returns `d` if `x > 0`, else `0.0`."""
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float], lst: Iterable[float]) -> list[float]:
    """Applies function fn to each element in lst."""
    return [fn(x) for x in lst]


def zipWith(fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]) -> list[float]:
    """Applies function fn to corresponding elements of lst1 and lst2."""
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(fn: Callable[[float, float], float], lst: Iterable[float], start: float) -> float:
    """Reduces lst using function fn starting with start value."""
    result = start
    for x in lst:
        result = fn(result, x)
    return result


def negList(lst: Iterable[float]) -> list[float]:
    """Negates each element in lst."""
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> list[float]:
    """Adds corresponding elements of lst1 and lst2."""
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Returns the sum of elements in lst."""
    return reduce(add, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    """Returns the product of elements in lst."""
    return reduce(mul, lst, 1.0)
