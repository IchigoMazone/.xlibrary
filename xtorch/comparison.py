from xtorch._utils import asarray, wrap, write_out

def eq(input, other, *, out=None):
    data = asarray(input) == asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def ne(input, other, *, out=None):
    data = asarray(input) != asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def gt(input, other, *, out=None):
    data = asarray(input) > asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def ge(input, other, *, out=None):
    data = asarray(input) >= asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def lt(input, other, *, out=None):
    data = asarray(input) < asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

def le(input, other, *, out=None):
    data = asarray(input) <= asarray(other)
    return write_out(out, data) or wrap(data, like=input, requires_grad=False)

