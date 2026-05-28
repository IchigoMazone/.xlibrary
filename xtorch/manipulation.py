from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out


def reshape(input, shape):
    return wrap(cp.reshape(asarray(input), shape), like=input)

def flatten(input, start_dim=0, end_dim=-1):
    data = asarray(input)
    ndim = data.ndim
    if end_dim < 0:
        end_dim += ndim
    if start_dim < 0:
        start_dim += ndim
    if start_dim > end_dim:
        raise ValueError("start_dim phai <= end_dim")
    flat_size = int(cp.prod(cp.asarray(data.shape[start_dim:end_dim + 1])).item())
    shape = data.shape[:start_dim] + (flat_size,) + data.shape[end_dim + 1:]
    return wrap(cp.reshape(data, shape), like=input)

def squeeze(input, dim=None):
    return wrap(cp.squeeze(asarray(input), axis=dim), like=input)

def unsqueeze(input, dim):
    return wrap(cp.expand_dims(asarray(input), axis=dim), like=input)

def transpose(input, dim0, dim1):
    data = asarray(input)
    axes = list(range(data.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return wrap(cp.transpose(data, axes), like=input)

def permute(input, dims):
    return wrap(cp.transpose(asarray(input), axes=dims), like=input)

def cat(tensors, dim=0, *, out=None):
    data = cp.concatenate([asarray(t) for t in tensors], axis=dim)
    like = tensors[0] if tensors else None
    return write_out(out, data) or wrap(data, like=like)

def stack(tensors, dim=0, *, out=None):
    data = cp.stack([asarray(t) for t in tensors], axis=dim)
    like = tensors[0] if tensors else None
    return write_out(out, data) or wrap(data, like=like)

def movedim(input, source, destination):
    return wrap(cp.moveaxis(asarray(input), source, destination), like=input)

def swapaxes(input, axis0, axis1):
    return wrap(cp.swapaxes(asarray(input), axis0, axis1), like=input)

def swapdims(input, dim0, dim1):
    return swapaxes(input, dim0, dim1)

def tile(input, reps):
    return wrap(cp.tile(asarray(input), reps), like=input)

def repeat(input, repeats, dim=None):
    return wrap(cp.repeat(asarray(input), repeats, axis=dim), like=input)

def expand(input, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = tuple(sizes[0])
    return wrap(cp.broadcast_to(asarray(input), sizes), like=input)

def broadcast_to(input, shape):
    return wrap(cp.broadcast_to(asarray(input), shape), like=input)

def flip(input, dims):
    return wrap(cp.flip(asarray(input), axis=dims), like=input)

def roll(input, shifts, dims=None):
    return wrap(cp.roll(asarray(input), shift=shifts, axis=dims), like=input)

def split(input, split_size_or_sections, dim=0):
    data = asarray(input)
    if isinstance(split_size_or_sections, int):
        indices = list(range(split_size_or_sections, data.shape[dim], split_size_or_sections))
        parts = cp.split(data, indices, axis=dim)
    else:
        indices = cp.cumsum(cp.asarray(split_size_or_sections))[:-1].tolist()
        parts = cp.split(data, indices, axis=dim)
    return tuple(wrap(part, like=input) for part in parts)

def chunk(input, chunks, dim=0):
    return tuple(wrap(part, like=input) for part in cp.array_split(asarray(input), chunks, axis=dim))

def unbind(input, dim=0):
    data = cp.moveaxis(asarray(input), dim, 0)
    return tuple(wrap(part, like=input) for part in data)
