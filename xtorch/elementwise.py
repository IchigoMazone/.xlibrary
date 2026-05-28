from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out

def neg(input, *, out=None):
    data = -asarray(input)
    return write_out(out, data) or wrap(data, like=input)

def abs(input, *, out=None):
    data = cp.abs(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def sqrt(input, *, out=None):
    data = cp.sqrt(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def rsqrt(input, *, out=None):
    data = 1 / cp.sqrt(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def square(input, *, out=None):
    data = cp.square(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def exp(input, *, out=None):
    data = cp.exp(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def log(input, *, out=None):
    data = cp.log(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def log2(input, *, out=None):
    data = cp.log2(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def log10(input, *, out=None):
    data = cp.log10(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def sin(input, *, out=None):
    data = cp.sin(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def cos(input, *, out=None):
    data = cp.cos(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def tan(input, *, out=None):
    data = cp.tan(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def asin(input, *, out=None):
    data = cp.arcsin(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def acos(input, *, out=None):
    data = cp.arccos(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def atan(input, *, out=None):
    data = cp.arctan(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def sinh(input, *, out=None):
    data = cp.sinh(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def cosh(input, *, out=None):
    data = cp.cosh(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def tanh(input, *, out=None):
    data = cp.tanh(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def floor(input, *, out=None):
    data = cp.floor(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def ceil(input, *, out=None):
    data = cp.ceil(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def round(input, *, decimals=0, out=None):
    data = cp.round(asarray(input), decimals=decimals)
    return write_out(out, data) or wrap(data, like=input)

def trunc(input, *, out=None):
    data = cp.trunc(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def sign(input, *, out=None):
    data = cp.sign(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def reciprocal(input, *, out=None):
    data = cp.reciprocal(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def pow(input, exponent, *, out=None):
    data = cp.power(asarray(input), asarray(exponent))
    return write_out(out, data) or wrap(data, like=input)

def maximum(input, other, *, out=None):
    data = cp.maximum(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input)

def minimum(input, other, *, out=None):
    data = cp.minimum(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input)

def clamp(input, min=None, max=None, *, out=None):
    data = asarray(input)
    if min is not None:
        data = cp.maximum(data, min)
    if max is not None:
        data = cp.minimum(data, max)
    return write_out(out, data) or wrap(data, like=input)

def clip(input, min=None, max=None, *, out=None):
    return clamp(input, min=min, max=max, out=out)

def isnan(input):
    return wrap(cp.isnan(asarray(input)), like=input, requires_grad=False)

def isinf(input):
    return wrap(cp.isinf(asarray(input)), like=input, requires_grad=False)

def isfinite(input):
    return wrap(cp.isfinite(asarray(input)), like=input, requires_grad=False)
