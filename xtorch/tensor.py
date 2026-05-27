from xtorch.multi_class import Size
import cupy as cp

class Tensor:

    def __init__(self, data, requires_grad=False, dtype=cp.float32):

        data = self._checkValueError(lambda: isinstance(data, Tensor), data)
        self.data = cp.asarray(data, dtype=dtype)
        self._check_requires_grad(requires_grad, dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._pred = set()
        self._op = ""
        self._backward = lambda: None

    def _check_requires_grad(self, requires_grad, dtype):
        if requires_grad:
            is_float = cp.issubdtype(dtype, cp.floating)
            is_complex = cp.issubdtype(dtype, cp.complexfloating)

            if not (is_float or is_complex):
                raise ValueError("requires_grad phai co kieu du lieu float hoac complex")

    def _checkTypeError(self, data, dtype):
        if not isinstance(data, dtype):
            raise ValueError(f"Du lieu phai co kieu du lieu {self.dtype}")
    
    def _checkValueError(self, callback, data):
        if callback: return data
        raise ValueError(f"Du lieu khong chinh xac")
    
    def __repr__(self):
        data_str = cp.array2string(
            self.data,
            separator=", ",
            precision=4,
            suppress_small=False,
            prefix="tensor("
        )

        if self.requires_grad:
            return f"tensor({data_str}, requires_grad=True)"
        return f"tensor({data_str})"
    
    @property
    def shape(self):
        return Size(self.data.shape)
    
    @property
    def dtype(self):
        return f"xtorch.{self.data.dtype}"
    
    @property
    def ndim(self):
        return len(self.data.shape)
    
    @staticmethod
    def _to_cupy_data(obj):

        if isinstance(obj, Tensor): 
            return obj.data
        
        if isinstance(obj, (int, float, cp.array)):
            return obj
        
        raise ValueError("Du lieu khong chinh xac")
    
    def __add__(self, other):
        other_data = Tensor._to_cupy_data(other) 
        return Tensor(self.data + other_data)
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(self.data - other_data)
    
    def __rsub__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(other_data - self.data)
    
    def __mul__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(self.data * other_data)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(self.data / other_data)
    
    def __rtruediv__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(cp.round(other_data / self.data, 4))
    
    def __matmul__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(self.data @ other_data)

    def __rmatmul__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(other_data @ self.data)

    def __pow__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(self.data ** other_data)

    def __rpow__(self, other):
        other_data = Tensor._to_cupy_data(other)
        return Tensor(other_data ** self.data)

    def sum(self, dim=None, keepdim=False):
        out_data = cp.sum(self.data, axis=dim, keepdims=keepdim)
        return Tensor(out_data, requires_grad=self.requires_grad)
    
    def mean(self, dim=None, keepdim=False):
        out_data = cp.mean(self.data, axis=dim, keepdims=keepdim)
        return Tensor(out_data, requires_grad=self.requires_grad)


def tensor(data, requires_grad=False, dtype=cp.float32):
    return Tensor(data, requires_grad, dtype)