from .activation import *
from .arithmetic import *
from .comparison import *
from .creation import *
from .elementwise import *
from . import linalg
from .logical import *
from .manipulation import *
from .matrix import *
from .random import *
from .reduction import *
from .tensor import *
from xtorch.backend import cp

float16 = cp.float16
float32 = cp.float32
float64 = cp.float64

int8 = cp.int8
int16 = cp.int16
int32 = cp.int32
int64 = cp.int64
uint8 = cp.uint8

complex64 = cp.complex64
complex128 = cp.complex128

bool = cp.bool_

for _name in ("asarray", "wrap", "write_out"):
    globals().pop(_name, None)
del _name
