from xtorch.backend import cp
from xtorch.tensor import Tensor
from xtorch._utils import asarray


def _shape_args(shape, *extra):
    if extra:
        return (shape,) + extra
    if isinstance(shape, (list, tuple)):
        return tuple(shape)
    return (shape,)


def tensor(data, dtype=None, device=None, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor) and (dtype is None or data.dtype == dtype):
        return data
    return Tensor(data, dtype=dtype, device=device)


def scalar_tensor(s, dtype=None, device=None, requires_grad=False):
    return Tensor(s, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    return Tensor(cp.zeros(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def ones(shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    return Tensor(cp.ones(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def empty(shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    return Tensor(cp.empty(shape, dtype=dtype), requires_grad=requires_grad, device=device)


def full(shape, fill_value, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    return Tensor(cp.full(shape, fill_value, dtype=dtype), requires_grad=requires_grad, device=device)


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        data = cp.arange(start, dtype=dtype)
    else:
        data = cp.arange(start, end, step, dtype=dtype)

    return Tensor(data, requires_grad=requires_grad, device=device)


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.linspace(start, end, steps, dtype=dtype), requires_grad=requires_grad, device=device)


def logspace(start, end, steps, base=10.0, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.logspace(start, end, steps, base=base, dtype=dtype), requires_grad=requires_grad, device=device)


def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.eye(n, M=m, dtype=dtype), requires_grad=requires_grad, device=device)


def empty_like(input, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.empty_like(asarray(input), dtype=dtype), requires_grad=requires_grad, device=device)

def zeros_like(input, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.zeros_like(asarray(input), dtype=dtype), requires_grad=requires_grad, device=device)

def ones_like(input, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.ones_like(asarray(input), dtype=dtype), requires_grad=requires_grad, device=device)

def full_like(input, fill_value, dtype=None, device=None, requires_grad=False):
    return Tensor(cp.full_like(asarray(input), fill_value, dtype=dtype), requires_grad=requires_grad, device=device)

def rand_like(input, dtype=None, device=None, requires_grad=False):
    data = cp.random.random_sample(asarray(input).shape)
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)

def randn_like(input, dtype=None, device=None, requires_grad=False):
    data = cp.random.standard_normal(asarray(input).shape)
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, requires_grad=requires_grad, device=device)


def from_numpy(ndarray):
    return Tensor(cp.asarray(ndarray))


def diag(input, diagonal=0, *, out=None):
    data = cp.diag(asarray(input), k=diagonal)
    if out is not None:
        out.data = data
        return out
    return Tensor(data)


def diagflat(input, offset=0):
    return Tensor(cp.diagflat(asarray(input), k=offset))
