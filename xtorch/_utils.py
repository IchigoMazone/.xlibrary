from xtorch.backend import cp

from xtorch.tensor import Tensor


def asarray(value, dtype=None):
    data = value.data if isinstance(value, Tensor) else value
    return cp.asarray(data, dtype=dtype)


def wrap(data, like=None, requires_grad=None, device=None):
    if requires_grad is None:
        requires_grad = bool(getattr(like, "requires_grad", False))
    if device is None:
        device = getattr(like, "device", None)
    return Tensor(data, requires_grad=requires_grad, device=device)


def write_out(out, data):
    if out is None:
        return None
    if not isinstance(out, Tensor):
        raise TypeError("out phai la Tensor")
    out.data = data
    return out
