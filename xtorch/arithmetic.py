
from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out


def add(input, other, *, alpha=1, out=None):
    data = asarray(input) + alpha * asarray(other)
    return write_out(out, data) or wrap(data, like=input)

def sub(input, other, *, alpha=1, out=None):
    data = asarray(input) - alpha * asarray(other)
    return write_out(out, data) or wrap(data, like=input)

def mul(input, other, *, out=None):
    data = asarray(input) * asarray(other)
    return write_out(out, data) or wrap(data, like=input)

def div(input, other, *, rounding_mode=None, out=None):
    data = asarray(input) / asarray(other)
    if rounding_mode == "floor":
        data = cp.floor(data)
    elif rounding_mode == "trunc":
        data = cp.trunc(data)
    elif rounding_mode is not None:
        raise ValueError("rounding_mode phai la None, 'floor' hoac 'trunc'")
    return write_out(out, data) or wrap(data, like=input)
