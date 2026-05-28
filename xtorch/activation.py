from xtorch.backend import cp

from xtorch._utils import asarray, wrap, write_out

def relu(input, *, out=None):
    data = cp.maximum(asarray(input), 0)
    return write_out(out, data) or wrap(data, like=input)

def sigmoid(input, *, out=None):
    data = 1 / (1 + cp.exp(-asarray(input)))
    return write_out(out, data) or wrap(data, like=input)

def tanh(input, *, out=None):
    data = cp.tanh(asarray(input))
    return write_out(out, data) or wrap(data, like=input)

def softmax(input, dim, *, out=None):
    data = asarray(input)
    shifted = data - cp.max(data, axis=dim, keepdims=True)
    exp_data = cp.exp(shifted)
    result = exp_data / cp.sum(exp_data, axis=dim, keepdims=True)
    return write_out(out, result) or wrap(result, like=input)

def log_softmax(input, dim, *, out=None):
    data = asarray(input)
    shifted = data - cp.max(data, axis=dim, keepdims=True)
    result = shifted - cp.log(cp.sum(cp.exp(shifted), axis=dim, keepdims=True))
    return write_out(out, result) or wrap(result, like=input)
