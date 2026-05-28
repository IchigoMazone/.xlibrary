from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out

def logical_and(input, other, *, out=None):
    data = cp.logical_and(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def logical_or(input, other, *, out=None):
    data = cp.logical_or(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def logical_xor(input, other, *, out=None):
    data = cp.logical_xor(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def logical_not(input, *, out=None):
    data = cp.logical_not(asarray(input))
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def where(condition, input=None, other=None):
    if input is None and other is None:
        return tuple(wrap(x, requires_grad=False) for x in cp.where(asarray(condition)))
    if input is None or other is None:
        raise ValueError("where can ca input va other, hoac khong co ca hai")
    data = cp.where(asarray(condition), asarray(input), asarray(other))
    return wrap(data, like=input)
