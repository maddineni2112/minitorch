from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import minitorch

from . import operators
from .autodiff import Context
import math

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction(ABC):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    @abstractmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    @abstractmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        return a + b

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a, b)
        return a * b

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        (a, b) = ctx.saved_values
        grad_a = d_output * b  # derivative of a * b with respect to a
        grad_b = d_output * a  # derivative of a * b with respect to b
        return grad_a, grad_b


class Inv(ScalarFunction):
    "Inverse function"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return 1.0 / a

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a = cast(float, ctx.saved_values[0])
        grad_a = -d_output / (a ** 2)
        return grad_a


class Neg(ScalarFunction):
    "Negation function"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return float(-a)

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        # Derivative of -a is -1
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        sig = 1 / (1 + math.exp(-a))  # Sigmoid function
        ctx.save_for_backward(a)
        return sig

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        sig = 1 / (1 + math.exp(-a))
        grad_a = sig * (1 - sig) * d_output  # derivative of sigmoid
        return grad_a


class ReLU(ScalarFunction):
    "ReLU function"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        relu = max(0.0, a)  # ReLU function: max(0, a)
        ctx.save_for_backward(a)
        return relu

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        grad_a = d_output if a > 0 else 0  # derivative of ReLU
        return grad_a


class Exp(ScalarFunction):
    "Exp function"

    @classmethod
    def forward(cls, ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        exp_val = math.exp(a)  # Exponential function
        ctx.save_for_backward(a)
        return exp_val

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        grad_a = math.exp(a) * d_output  # derivative of exp(a)
        return grad_a


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        result = 1.0 if a < b else 0.0
        ctx.save_for_backward(a, b)
        return result

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0  # Derivatives are zero since LT is non-differentiable


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @classmethod
    def forward(cls, ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        result = 1.0 if a == b else 0.0
        ctx.save_for_backward(a, b)
        return result

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0  # Derivatives are zero since EQ is non-differentiable
