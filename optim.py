from jax import value_and_grad
import jax


class SGD:
    def __init__(self, loss_fn, lr=0.1):
        self.lr = lr
        self.grad_loss_fn = value_and_grad(loss_fn, has_aux=True)

    def update(self, params, x, t, cached_values=None):
        (v, cached_values), g = self.grad_loss_fn(
            params, x, t, cached_values=cached_values)
        new_params = jax.tree_map(lambda p, g: p - self.lr * g, params, g)
        return v, new_params, cached_values
