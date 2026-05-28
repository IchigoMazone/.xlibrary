from xtorch.backend import cp

class Tensor:

    def __init__(self, data, requires_grad=False):
        self.data = cp.array(data)
        self.requires_grad = requires_grad
        self.grad = None

        