from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out

def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.sum(asarray(input), axis=dim, keepdims=keepdim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.mean(asarray(input), axis=dim, keepdims=keepdim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def prod(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.prod(asarray(input), axis=dim, keepdims=keepdim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def max(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.max(asarray(input).astype(dtype) if dtype is not None else asarray(input), axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def min(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.min(asarray(input).astype(dtype) if dtype is not None else asarray(input), axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def argmax(input, dim=None, keepdim=False):
    data = cp.argmax(asarray(input), axis=dim)
    if keepdim and dim is not None:
        data = cp.expand_dims(data, axis=dim)
    return wrap(data, like=input, requires_grad=False)

def argmin(input, dim=None, keepdim=False):
    data = cp.argmin(asarray(input), axis=dim)
    if keepdim and dim is not None:
        data = cp.expand_dims(data, axis=dim)
    return wrap(data, like=input, requires_grad=False)

def var(input, dim=None, *, correction=1, keepdim=False, out=None):
    data = cp.var(asarray(input), axis=dim, ddof=correction, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    data = cp.std(asarray(input), axis=dim, ddof=correction, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def nansum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    data = cp.nansum(asarray(input), axis=dim, keepdims=keepdim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def all(input, dim=None, keepdim=False, *, out=None):
    data = cp.all(asarray(input), axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def any(input, dim=None, keepdim=False, *, out=None):
    data = cp.any(asarray(input), axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def median(input, dim=None, keepdim=False, *, out=None):
    data = cp.median(asarray(input), axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def quantile(input, q, dim=None, keepdim=False, *, out=None):
    data = cp.quantile(asarray(input), q, axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=input)

def cumsum(input, dim, *, dtype=None, out=None):
    data = cp.cumsum(asarray(input), axis=dim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def cumprod(input, dim, *, dtype=None, out=None):
    data = cp.cumprod(asarray(input), axis=dim, dtype=dtype)
    return write_out(out, data) or wrap(data, like=input)

def trace(input, *, out=None):
    data = cp.trace(asarray(input))
    return write_out(out, data) or wrap(data, like=input)
