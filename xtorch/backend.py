try:
    import cupy as _cp

    try:
        _cp.cuda.runtime.getDeviceCount()
        cp = _cp
        HAS_CUDA = True
    except Exception:
        import numpy as cp

        HAS_CUDA = False
except ModuleNotFoundError:
    import numpy as cp

    HAS_CUDA = False


if not hasattr(cp, "asnumpy"):
    cp.asnumpy = cp.asarray
