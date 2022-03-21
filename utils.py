import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from jax.lax import conv_with_general_padding, reduce_window
import jax.lax as lax
import tqdm

"""
Activation Functions
"""


@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


@jit
def tanh(x):
    return jnp.tanh(x)


@jit
def relu(x):
    return jnp.maximum(x, 0.0)


@jit
def leaky_relu(x, alpha=0.1):
    return jnp.maximum(x, alpha*x)


@jit
def silu(x):
    return x * sigmoid(x)


@jit
def gelu(x):
    """
    Approximated form of GELU
    """
    return x * sigmoid(1.702*x)


@jit
@vmap
def softmax(x):
    e = jnp.exp(x)
    return e / jnp.sum(e)


@jit
def logsoftmax(x):
    return jnp.log(softmax(x))


"""
Neural Network operations
"""


@jit
def addmm(x, W, b=None):
    if b is None:
        return x @ W
    return x @ W + b


def conv1d(x, W, b=None, strides=(1,), paddings=((0, 0),)):
    o = conv_with_general_padding(x, W, strides, paddings, (1,), (1,))
    if b is not None:
        o = o + b
    return o


def conv2d(x, W, b=None, strides=(1, 1), paddings=((0, 0), (0, 0))):
    o = conv_with_general_padding(x, W, strides, paddings, (1, 1), (1, 1))
    if b is not None:
        o = o + b
    return o


def max_pool1d(x, kernel_size=(2,)):
    o = reduce_window(x, -jnp.inf, lax.max,
                      (1, 1, *kernel_size), (1, 1, *kernel_size), ((0, 0), (0, 0), (0, 0)))
    return o


def max_pool2d(x, kernel_size=(2, 2)):
    o = reduce_window(x, -jnp.inf, lax.max,
                      (1, 1, *kernel_size), (1, 1, *kernel_size), ((0, 0), (0, 0), (0, 0), (0, 0)))
    return o


@jit
def batch_norm1d(x, gamma, beta, running_mean, running_var, momentum=0.1, training=True):
    mu_ = running_mean
    var_ = running_var
    if training:
        N, C, L = x.shape
        mu_ = jnp.mean(x, axis=(0, 2)).reshape(1, C, 1)
        var_ = jnp.var(x, axis=(0, 2)).reshape(1, C, 1)
        running_mean = mu_ * momentum + running_mean * (1 - momentum)
        running_var = var_ * momentum + running_var * (1 - momentum)

    o = (x - mu_)/(var_ + 1e-5) ** 0.5
    o = o * gamma + beta
    return o, running_mean, running_var


@jit
def batch_norm2d(x, gamma, beta, running_mean, running_var, momentum=0.1, training=True):
    mu_ = running_mean
    var_ = running_var
    if training:
        N, C, H, W = x.shape
        mu_ = jnp.mean(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        var_ = jnp.var(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        running_mean_ = mu_ * momentum + running_mean * (1 - momentum)
        running_var_ = var_ * momentum + running_var * (1 - momentum)

    o = (x - mu_)/(var_ + 1e-5) ** 0.5
    o = o * gamma + beta
    return o, running_mean_, running_var_


@jit
def dropout(x, p=0.1, training=True):
    if training:
        mask = (np.random.random(x.shape) > p) * 1/p
        return x * mask
    return x


"""
Loss Functions
"""


@jit
def mse_loss(o, t):
    return ((o - t) ** 2).mean()


@jit
def nll_loss(o, t):
    return -jnp.sum(o * t)


"""
Datasets
"""


class Dataset:
    def __init__(self, x, y, transform=None, target_transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        assert len(self.x) == len(
            self.y), "x and y have different number of data"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class Dataloader:
    def __init__(self, ds, batch_size=32):
        self.ds = ds
        self.batch_size = batch_size
        self.num_batches = round(len(self.ds) / batch_size)
        self.curr = 0

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.curr >= len(self.ds):
            raise StopIteration
        tmp = self.curr
        self.curr += self.batch_size
        self.curr = min(self.curr, len(self.ds))
        batch_x = []
        batch_y = []
        for i in range(tmp, self.curr):
            x_, y_ = self.ds[i]
            batch_x.append(x_)
            batch_y.append(y_)
        return np.vstack(batch_x), np.vstack(batch_y)

    def __iter__(self):
        self.curr = 0
        return self


"""
Training and Evaluating utilites
"""


def train(params, optim, train_dl, cached_values=None, epochs=10):
    for epoch in tqdm.tqdm(range(epochs)):
        tot_loss = 0
        for x, y in train_dl:
            loss_, params, cached_values = optim.update(
                params, x, y, cached_values=cached_values)
            tot_loss += loss_
        print(f"epoch: {epoch}, loss: {tot_loss / len(train_dl)}")
    return params, cached_values
