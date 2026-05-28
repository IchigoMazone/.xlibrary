from xtorch.tensor import Tensor
from xtorch.backend import cp


def _shape_args(shape, *extra):
    if extra:
        return (shape,) + extra
    if isinstance(shape, (list, tuple)):
        return tuple(shape)
    return (shape,)


def manual_seed(seed):
    cp.random.seed(seed)


def rand(shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    data = cp.random.rand(*shape)

    if dtype is not None:
        data = data.astype(dtype)

    return Tensor(
        data,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def randn(shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    data = cp.random.randn(*shape)

    if dtype is not None:
        data = data.astype(dtype)

    return Tensor(
        data,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def randint(low, high, shape, *extra, dtype=None, device=None, requires_grad=False):
    shape = _shape_args(shape, *extra)
    data = cp.random.randint(low, high, size=shape)

    if dtype is not None:
        data = data.astype(dtype)

    return Tensor(
        data,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def randperm(n, dtype=None, device=None, requires_grad=False):
    data = cp.random.permutation(n)

    if dtype is not None:
        data = data.astype(dtype)

    return Tensor(
        data,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def normal(mean=0.0, std=1.0, size=None, dtype=None, device=None, requires_grad=False):
    data = cp.random.normal(mean, std, size=size)
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def uniform(low=0.0, high=1.0, size=None, dtype=None, device=None, requires_grad=False):
    data = cp.random.uniform(low, high, size=size)
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
