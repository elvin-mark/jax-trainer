from functools import cache
from jax import grad, value_and_grad


class SGD:
    def __init__(self, loss_fn, lr=0.1):
        self.lr = lr
        self.grad_loss_fn = value_and_grad(loss_fn, has_aux=True)

    def update(self, params, x, t, cached_values=None):
        (v, cached_values), g = self.grad_loss_fn(
            params, x, t, cached_values=cached_values)
        new_params = {}
        for l in params:
            new_params[l] = {}
            for k in params[l]:
                new_params[l][k] = params[l][k] - self.lr * g[l][k]
        return v, new_params, cached_values
