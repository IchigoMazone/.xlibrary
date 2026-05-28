from xtorch.multi_class import Size
from xtorch.backend import HAS_CUDA, cp


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None, device=None):
        if isinstance(data, Tensor):
            data = data.data

        source_is_array = isinstance(data, cp.ndarray)
        self._data = cp.asarray(data, dtype=dtype)
        if dtype is None and not source_is_array:
            if self._data.dtype.kind == "f":
                self._data = self._data.astype(cp.float32)
            elif self._data.dtype.kind == "c":
                self._data = self._data.astype(cp.complex64)
        self._requires_grad = bool(requires_grad)
        self._device = device or ("cuda" if HAS_CUDA else "cpu")
        self._grad = None
        self._pred = set()
        self._op = ""
        self._backward = lambda: None

        self._check_requires_grad()

    def _check_requires_grad(self):
        if not self._requires_grad:
            return

        is_float = cp.issubdtype(self._data.dtype, cp.floating)
        is_complex = cp.issubdtype(self._data.dtype, cp.complexfloating)
        if not (is_float or is_complex):
            raise ValueError("requires_grad phai co kieu du lieu float hoac complex")

    def __repr__(self):
        data_str = cp.array2string(
            self._data,
            separator=", ",
            precision=4,
            suppress_small=False,
            prefix="tensor(",
        )
        default_dtypes = {cp.dtype(cp.float32), cp.dtype(cp.int64), cp.dtype(cp.bool_)}
        suffix = "" if self._data.dtype in default_dtypes else f", dtype=xtorch.{self.dtype}"
        if self.requires_grad:
            suffix += ", requires_grad=True"
        return f"tensor({data_str}{suffix})"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = cp.asarray(value)

    @property
    def shape(self):
        return Size(self._data.shape)

    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad, device=self.device)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = bool(value)
        self._check_requires_grad()

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @staticmethod
    def _to_cupy_data(obj):
        if isinstance(obj, Tensor):
            return obj.data
        return cp.asarray(obj)

    @staticmethod
    def _wrap(data, like=None, requires_grad=None):
        if requires_grad is None:
            requires_grad = bool(getattr(like, "requires_grad", False))
        device = getattr(like, "device", None)
        return Tensor(data, requires_grad=requires_grad, device=device)

    def _preserve_default_float(self, data, other):
        if not isinstance(other, Tensor) and self.data.dtype == cp.dtype(cp.float32) and data.dtype.kind == "f":
            return data.astype(cp.float32)
        return data

    def __add__(self, other):
        data = self.data + Tensor._to_cupy_data(other)
        return Tensor(self._preserve_default_float(data, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        data = self.data - Tensor._to_cupy_data(other)
        return Tensor(self._preserve_default_float(data, other))

    def __rsub__(self, other):
        data = Tensor._to_cupy_data(other) - self.data
        return Tensor(self._preserve_default_float(data, other))

    def __mul__(self, other):
        data = self.data * Tensor._to_cupy_data(other)
        return Tensor(self._preserve_default_float(data, other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        data = self.data / Tensor._to_cupy_data(other)
        return Tensor(self._preserve_default_float(data, other))

    def __rtruediv__(self, other):
        data = Tensor._to_cupy_data(other) / self.data
        return Tensor(self._preserve_default_float(data, other))

    def __matmul__(self, other):
        return Tensor(self.data @ Tensor._to_cupy_data(other))

    def __rmatmul__(self, other):
        return Tensor(Tensor._to_cupy_data(other) @ self.data)

    def __pow__(self, other):
        data = self.data ** Tensor._to_cupy_data(other)
        return Tensor(self._preserve_default_float(data, other))

    def __rpow__(self, other):
        data = Tensor._to_cupy_data(other) ** self.data
        return Tensor(self._preserve_default_float(data, other))

    def __neg__(self):
        return Tensor(-self.data)

    def __eq__(self, other):
        return Tensor(self.data == Tensor._to_cupy_data(other))

    def __ne__(self, other):
        return Tensor(self.data != Tensor._to_cupy_data(other))

    def __gt__(self, other):
        return Tensor(self.data > Tensor._to_cupy_data(other))

    def __ge__(self, other):
        return Tensor(self.data >= Tensor._to_cupy_data(other))

    def __lt__(self, other):
        return Tensor(self.data < Tensor._to_cupy_data(other))

    def __le__(self, other):
        return Tensor(self.data <= Tensor._to_cupy_data(other))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return Tensor(self.data[key], requires_grad=self.requires_grad, device=self.device)

    def __setitem__(self, key, value):
        self.data[key] = Tensor._to_cupy_data(value)

    def item(self):
        return self.data.item()

    def numpy(self):
        return cp.asnumpy(self.data)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        return int(self.data.size)

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self.device)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    def to(self, dtype=None, device=None):
        data = self.data.astype(dtype) if dtype is not None else self.data.copy()
        return Tensor(data, requires_grad=self.requires_grad, device=device or self.device)

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self):
        return self.to(device="cuda")

    def astype(self, dtype):
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad, device=self.device)

    def float(self):
        return self.astype(cp.float32)

    def double(self):
        return self.astype(cp.float64)

    def long(self):
        return self.astype(cp.int64)

    def int(self):
        return self.astype(cp.int32)

    def bool(self):
        return self.astype(cp.bool_)

    def sum(self, dim=None, keepdim=False, dtype=None):
        return Tensor(cp.sum(self.data, axis=dim, keepdims=keepdim, dtype=dtype), requires_grad=self.requires_grad, device=self.device)

    def mean(self, dim=None, keepdim=False, dtype=None):
        return Tensor(cp.mean(self.data, axis=dim, keepdims=keepdim, dtype=dtype), requires_grad=self.requires_grad, device=self.device)

    def prod(self, dim=None, keepdim=False, dtype=None):
        return Tensor(cp.prod(self.data, axis=dim, keepdims=keepdim, dtype=dtype), requires_grad=self.requires_grad, device=self.device)

    def max(self, dim=None, keepdim=False):
        return Tensor(cp.max(self.data, axis=dim, keepdims=keepdim), requires_grad=self.requires_grad, device=self.device)

    def min(self, dim=None, keepdim=False):
        return Tensor(cp.min(self.data, axis=dim, keepdims=keepdim), requires_grad=self.requires_grad, device=self.device)

    def argmax(self, dim=None, keepdim=False):
        data = cp.argmax(self.data, axis=dim)
        if keepdim and dim is not None:
            data = cp.expand_dims(data, axis=dim)
        return Tensor(data, requires_grad=False, device=self.device)

    def argmin(self, dim=None, keepdim=False):
        data = cp.argmin(self.data, axis=dim)
        if keepdim and dim is not None:
            data = cp.expand_dims(data, axis=dim)
        return Tensor(data, requires_grad=False, device=self.device)

    def std(self, dim=None, correction=1, keepdim=False):
        return Tensor(cp.std(self.data, axis=dim, ddof=correction, keepdims=keepdim), requires_grad=self.requires_grad, device=self.device)

    def var(self, dim=None, correction=1, keepdim=False):
        return Tensor(cp.var(self.data, axis=dim, ddof=correction, keepdims=keepdim), requires_grad=self.requires_grad, device=self.device)

    def all(self, dim=None, keepdim=False):
        return Tensor(cp.all(self.data, axis=dim, keepdims=keepdim), requires_grad=False, device=self.device)

    def any(self, dim=None, keepdim=False):
        return Tensor(cp.any(self.data, axis=dim, keepdims=keepdim), requires_grad=False, device=self.device)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(cp.reshape(self.data, shape), requires_grad=self.requires_grad, device=self.device)

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1):
        ndim = self.ndim
        if end_dim < 0:
            end_dim += ndim
        if start_dim < 0:
            start_dim += ndim
        flat_size = int(cp.prod(cp.asarray(self.data.shape[start_dim:end_dim + 1])).item())
        shape = self.data.shape[:start_dim] + (flat_size,) + self.data.shape[end_dim + 1:]
        return self.reshape(shape)

    def squeeze(self, dim=None):
        return Tensor(cp.squeeze(self.data, axis=dim), requires_grad=self.requires_grad, device=self.device)

    def unsqueeze(self, dim):
        return Tensor(cp.expand_dims(self.data, axis=dim), requires_grad=self.requires_grad, device=self.device)

    def transpose(self, dim0, dim1):
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(cp.transpose(self.data, axes), requires_grad=self.requires_grad, device=self.device)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return Tensor(cp.transpose(self.data, axes=dims), requires_grad=self.requires_grad, device=self.device)

    def repeat(self, repeats, dim=None):
        return Tensor(cp.repeat(self.data, repeats, axis=dim), requires_grad=self.requires_grad, device=self.device)

    def expand(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = tuple(sizes[0])
        return Tensor(cp.broadcast_to(self.data, sizes), requires_grad=self.requires_grad, device=self.device)

    def matmul(self, other):
        return self.__matmul__(other)

    def mm(self, mat2):
        if self.ndim != 2 or Tensor._to_cupy_data(mat2).ndim != 2:
            raise ValueError("mm chi ho tro tensor 2D")
        return self.__matmul__(mat2)

    def abs(self):
        return Tensor(cp.abs(self.data), requires_grad=self.requires_grad, device=self.device)

    def sqrt(self):
        return Tensor(cp.sqrt(self.data), requires_grad=self.requires_grad, device=self.device)

    def exp(self):
        return Tensor(cp.exp(self.data), requires_grad=self.requires_grad, device=self.device)

    def log(self):
        return Tensor(cp.log(self.data), requires_grad=self.requires_grad, device=self.device)

    def sin(self):
        return Tensor(cp.sin(self.data), requires_grad=self.requires_grad, device=self.device)

    def cos(self):
        return Tensor(cp.cos(self.data), requires_grad=self.requires_grad, device=self.device)

    def tanh(self):
        return Tensor(cp.tanh(self.data), requires_grad=self.requires_grad, device=self.device)

    def clamp(self, min=None, max=None):
        data = self.data
        if min is not None:
            data = cp.maximum(data, min)
        if max is not None:
            data = cp.minimum(data, max)
        return Tensor(data, requires_grad=self.requires_grad, device=self.device)
