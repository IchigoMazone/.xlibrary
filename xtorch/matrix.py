from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out


def matmul(input, other, *, out=None):
    data = cp.matmul(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input)


def mm(input, mat2, out_dtype=None, *, out=None):
    a = asarray(input)
    b = asarray(mat2)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm chi ho tro tensor 2D")
    data = cp.matmul(a, b)
    if out_dtype is not None:
        data = data.astype(out_dtype)
    return write_out(out, data) or wrap(data, like=input)


def bmm(input, mat2, *, out=None):
    a = asarray(input)
    b = asarray(mat2)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("bmm chi ho tro tensor 3D")
    data = cp.matmul(a, b)
    return write_out(out, data) or wrap(data, like=input)


def mv(input, vec, *, out=None):
    a = asarray(input)
    b = asarray(vec)
    if a.ndim != 2 or b.ndim != 1:
        raise ValueError("mv can input 2D va vec 1D")
    data = cp.matmul(a, b)
    return write_out(out, data) or wrap(data, like=input)


def dot(input, tensor, *, out=None):
    data = cp.dot(asarray(input), asarray(tensor))
    return write_out(out, data) or wrap(data, like=input)


def outer(input, vec2, *, out=None):
    data = cp.outer(asarray(input), asarray(vec2))
    return write_out(out, data) or wrap(data, like=input)


def tensordot(a, b, dims=2, *, out=None):
    data = cp.tensordot(asarray(a), asarray(b), axes=dims)
    return write_out(out, data) or wrap(data, like=a)
