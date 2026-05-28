from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out


def _linalg(name):
    return getattr(cp.linalg, name)


def _maybe_tuple(result, like=None, requires_grad=None):
    if isinstance(result, tuple):
        return tuple(wrap(value, like=like, requires_grad=requires_grad) for value in result)
    return wrap(result, like=like, requires_grad=requires_grad)


def matmul(input, other, *, out=None):
    data = cp.matmul(asarray(input), asarray(other))
    return write_out(out, data) or wrap(data, like=input)


def inv(A, *, out=None):
    data = _linalg("inv")(asarray(A))
    return write_out(out, data) or wrap(data, like=A)


def pinv(A, *, rtol=None, hermitian=False, out=None):
    kwargs = {"hermitian": hermitian}
    if rtol is not None:
        kwargs["rcond"] = rtol
    data = _linalg("pinv")(asarray(A), **kwargs)
    return write_out(out, data) or wrap(data, like=A)


def solve(A, B, *, left=True, out=None):
    if not left:
        data = cp.swapaxes(_linalg("solve")(cp.swapaxes(asarray(A), -1, -2), cp.swapaxes(asarray(B), -1, -2)), -1, -2)
    else:
        data = _linalg("solve")(asarray(A), asarray(B))
    return write_out(out, data) or wrap(data, like=B)


def det(A, *, out=None):
    data = _linalg("det")(asarray(A))
    return write_out(out, data) or wrap(data, like=A)


def slogdet(A, *, out=None):
    result = _linalg("slogdet")(asarray(A))
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out cua slogdet phai la tuple gom 2 Tensor")
        write_out(out[0], result[0])
        write_out(out[1], result[1])
        return out
    return _maybe_tuple(result, like=A)


def eig(A, *, out=None):
    result = _linalg("eig")(asarray(A))
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out cua eig phai la tuple gom 2 Tensor")
        write_out(out[0], result[0])
        write_out(out[1], result[1])
        return out
    return _maybe_tuple(result, like=A)


def eigvals(A, *, out=None):
    data = _linalg("eigvals")(asarray(A))
    return write_out(out, data) or wrap(data, like=A)


def eigh(A, UPLO="L", *, out=None):
    result = _linalg("eigh")(asarray(A), UPLO=UPLO)
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out cua eigh phai la tuple gom 2 Tensor")
        write_out(out[0], result[0])
        write_out(out[1], result[1])
        return out
    return _maybe_tuple(result, like=A)


def eigvalsh(A, UPLO="L", *, out=None):
    data = _linalg("eigvalsh")(asarray(A), UPLO=UPLO)
    return write_out(out, data) or wrap(data, like=A)


def svd(A, full_matrices=True, *, driver=None, out=None):
    result = _linalg("svd")(asarray(A), full_matrices=full_matrices)
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 3:
            raise TypeError("out cua svd phai la tuple gom 3 Tensor")
        write_out(out[0], result[0])
        write_out(out[1], result[1])
        write_out(out[2], result[2])
        return out
    return _maybe_tuple(result, like=A)


def svdvals(A, *, driver=None, out=None):
    data = _linalg("svd")(asarray(A), compute_uv=False)
    return write_out(out, data) or wrap(data, like=A)


def qr(A, mode="reduced", *, out=None):
    result = _linalg("qr")(asarray(A), mode=mode)
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out cua qr phai la tuple gom 2 Tensor")
        write_out(out[0], result[0])
        write_out(out[1], result[1])
        return out
    return _maybe_tuple(result, like=A)


def cholesky(A, *, upper=False, out=None):
    data = _linalg("cholesky")(asarray(A))
    if upper:
        data = cp.swapaxes(data, -1, -2).conj()
    return write_out(out, data) or wrap(data, like=A)


def matrix_power(A, n, *, out=None):
    data = _linalg("matrix_power")(asarray(A), n)
    return write_out(out, data) or wrap(data, like=A)


def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
    kwargs = {"hermitian": hermitian}
    if atol is not None:
        kwargs["tol"] = atol
    if rtol is not None and atol is None:
        kwargs["tol"] = rtol
    data = _linalg("matrix_rank")(asarray(A), **kwargs)
    return write_out(out, data) or wrap(data, like=A, requires_grad=False)


def cond(A, p=None, *, out=None):
    data = _linalg("cond")(asarray(A), p=p)
    return write_out(out, data) or wrap(data, like=A)


def norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    data = _linalg("norm")(asarray(A, dtype=dtype), ord=ord, axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=A)


def vector_norm(x, ord=2, dim=None, keepdim=False, *, out=None, dtype=None):
    data = asarray(x, dtype=dtype)
    if dim is None:
        data = data.reshape(-1)
    result = _linalg("norm")(data, ord=ord, axis=dim, keepdims=keepdim)
    return write_out(out, result) or wrap(result, like=x)


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False, *, out=None, dtype=None):
    data = _linalg("norm")(asarray(A, dtype=dtype), ord=ord, axis=dim, keepdims=keepdim)
    return write_out(out, data) or wrap(data, like=A)


def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    data = cp.diagonal(asarray(A), offset=offset, axis1=dim1, axis2=dim2)
    return wrap(data, like=A)
