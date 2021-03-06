from functools import partial
from jax import random, jit
from utils import addmm, batch_norm1d, batch_norm2d, conv1d, conv2d, dropout, gelu, leaky_relu, max_pool2d, relu, sigmoid, silu, tanh, max_pool1d
import numpy as np
from jax.lax import scan


class Sigmoid:
    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return sigmoid(x), cached_values


class Tanh:
    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return tanh(x), cached_values


class ReLU:
    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return relu(x), cached_values


class LeakyReLU:
    def __init__(self, alpha):
        self.alpha = alpha

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return leaky_relu(x, alpha=self.alpha), cached_values


class SiLU:
    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return silu(x), cached_values


class GELU:
    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return gelu(x), cached_values


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = True

    def init_params(self, key):
        params = {}
        cached_values = None
        key_W, key_b = random.split(key)
        params["W"] = random.normal(
            key_W, (self.in_features, self.out_features)) / self.in_features ** 0.5
        if self.bias:
            params["b"] = random.normal(
                key_b, (self.out_features,)) / self.in_features ** 0.5
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        W = params["W"]
        b = None if "b" not in params else params["b"]
        return addmm(x, W, b=b), cached_values


class Conv1d:
    def __init__(self, in_planes, out_planes, bias=True, kernel_size=3, stride=1, padding=0):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bias = bias
        self.kernel_size = kernel_size
        self.strides = (stride, )
        self.paddings = ((padding, padding), )

        @jit
        def conv_(x, W):
            return conv1d(x, W, strides=self.strides, paddings=self.paddings)
        self.conv_ = conv_

    def init_params(self, key):
        params = {}
        cached_values = None
        key_W, key_b = random.split(key)
        params["W"] = random.normal(key_W, (self.out_planes, self.in_planes,
                                    self.kernel_size)) / self.in_planes ** 0.5
        if self.bias:
            params["b"] = random.normal(
                key_b, (1, self.out_planes, 1)) / self.in_planes ** 0.5
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        W = params["W"]
        b = None if "b" not in params else params["b"]
        o = self.conv_(x, W)
        if b is not None:
            o = o + b
        return o, cached_values


class Conv2d:
    def __init__(self, in_planes, out_planes, bias=True, kernel_size=3, stride=1, padding=0):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.bias = bias
        self.kernel_size = kernel_size
        self.strides = (stride, stride)
        self.paddings = ((padding, padding), (padding, padding))

        @jit
        def conv_(x, W):
            return conv2d(x, W, strides=self.strides, paddings=self.paddings)
        self.conv_ = conv_

    def init_params(self, key):
        params = {}
        cached_values = None
        key_W, key_b = random.split(key)
        params["W"] = random.normal(key_W, (self.out_planes, self.in_planes,
                                    self.kernel_size, self.kernel_size)) / self.in_planes ** 0.5
        if self.bias:
            params["b"] = random.normal(
                key_b, (1, self.out_planes, 1, 1)) / self.in_planes ** 0.5
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        W = params["W"]
        b = None if "b" not in params else params["b"]
        o = self.conv_(x, W)
        if b is not None:
            o = o + b
        return o, cached_values


class MaxPool1d:
    def __init__(self, kernel_size=2):
        self.window = (kernel_size, )

        @jit
        def max_pool_(x):
            return max_pool1d(x, kernel_size=self.window)

        self.max_pool_ = max_pool_

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return self.max_pool_(x), cached_values


class MaxPool2d:
    def __init__(self, kernel_size=2):
        self.window = (kernel_size, kernel_size)

        @jit
        def max_pool_(x):
            return max_pool2d(x, kernel_size=self.window)

        self.max_pool_ = max_pool_

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return self.max_pool_(x), cached_values


class BatchNorm1d:
    def __init__(self, out_planes, momentum=0.1):
        self.out_planes = out_planes
        self.momentum = momentum

    def init_params(self, key):
        params = {}
        cached_values = {}
        params["gamma"] = random.normal(key, (1, self.out_planes, 1))
        params["beta"] = random.normal(key, (1, self.out_planes, 1))
        cached_values["running_mean"] = np.zeros((1, self.out_planes, 1))
        cached_values["running_var"] = np.ones((1, self.out_planes, 1))
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        gamma = params["gamma"]
        beta = params["beta"]
        rm = cached_values["running_mean"]
        rv = cached_values["running_var"]
        o, rm, rv = batch_norm1d(
            x, gamma, beta, rm, rv, momentum=self.momentum)
        cached_values = {}
        cached_values["running_mean"] = rm
        cached_values["running_var"] = rv
        return o, cached_values


class BatchNorm2d:
    def __init__(self, out_planes, momentum=0.1):
        self.out_planes = out_planes
        self.momentum = momentum

    def init_params(self, key):
        params = {}
        cached_values = {}
        params["gamma"] = random.normal(key, (1, self.out_planes, 1, 1))
        params["beta"] = random.normal(key, (1, self.out_planes, 1, 1))
        cached_values["running_mean"] = np.zeros((1, self.out_planes, 1, 1))
        cached_values["running_var"] = np.ones((1, self.out_planes, 1, 1))
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        gamma = params["gamma"]
        beta = params["beta"]
        rm = cached_values["running_mean"]
        rv = cached_values["running_var"]
        o, rm, rv = batch_norm2d(
            x, gamma, beta, rm, rv, momentum=self.momentum)
        cached_values = {}
        cached_values["running_mean"] = rm
        cached_values["running_var"] = rv
        return o, cached_values


class Dropout1d:
    def __init__(self, p=0.1):
        self.p = p

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        o = dropout(x, p=self.p)
        return o, cached_values


class Dropout2d:
    def __init__(self, p=0.1):
        self.p = p

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        o = dropout(x, p=self.p)
        return o, cached_values


class Flatten:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def init_params(self, key):
        return {}, {}

    def apply(self, params, x, cached_values=None):
        return x.reshape((self.new_shape)), cached_values


class Sequential:
    def __init__(self, *args):
        self.modules = [elem for elem in args]

    def init_params(self, key):
        params = {}
        cached_values = {}
        for i, m in enumerate(self.modules):
            layer_ = str(i)
            p, c = m.init_params(key)
            params[layer_] = p
            cached_values[layer_] = c

        return params, cached_values

    def apply(self, params, x, cached_values=None):
        o = x
        cached_values_ = {}
        for i, m in enumerate(self.modules):
            layer_ = str(i)
            p = params[layer_]
            c = cached_values[layer_]
            o, c = m.apply(p, o, c)
            cached_values_[layer_] = c
        return o, cached_values_


class BasicResidualBlock2d:
    def __init__(self, in_planes, out_planes):
        self.direct = Sequential(
            Conv2d(in_planes, out_planes, bias=False,
                   kernel_size=3, padding=1),
            BatchNorm2d(out_planes),
            ReLU(),
            Conv2d(out_planes, out_planes, bias=False, kernel_size=1),
            BatchNorm2d(out_planes)
        )
        self.relu = ReLU()
        self.shortcut = Sequential()
        if in_planes != out_planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, out_planes, bias=False, kernel_size=1),
                BatchNorm2d(out_planes)
            )

    def init_params(self, key):
        params = {}
        cached_values = {}
        params["direct"], cached_values["direct"] = self.direct.init_params(
            key)
        params["relu"], cached_values["relu"] = self.relu.init_params(key)
        params["shortcut"], cached_values["shortcut"] = self.shortcut.init_params(
            key)
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        cached_values_ = {}
        o, cached_values_["direct"] = self.direct.apply(
            params["direct"], x, cached_values=cached_values["direct"])

        i, cached_values_["shortcut"] = self.shortcut.apply(params["shortcut"], x,
                                                            cached_values=cached_values["shortcut"])

        o = o + i
        o, cached_values_["relu"] = self.relu.apply(
            params["relu"], o, cached_values=cached_values["relu"])
        return o, cached_values_


class BasicResidualBlock1d:
    def __init__(self, in_planes, out_planes):
        self.direct = Sequential(
            Conv1d(in_planes, out_planes, bias=False,
                   kernel_size=3, padding=1),
            BatchNorm1d(out_planes),
            ReLU(),
            Conv1d(out_planes, out_planes, bias=False, kernel_size=1),
            BatchNorm1d(out_planes)
        )
        self.relu = ReLU()
        self.shortcut = Sequential()
        if in_planes != out_planes:
            self.shortcut = Sequential(
                Conv1d(in_planes, out_planes, bias=False, kernel_size=1),
                BatchNorm1d(out_planes)
            )

    def init_params(self, key):
        params = {}
        cached_values = {}
        params["direct"], cached_values["direct"] = self.direct.init_params(
            key)
        params["relu"], cached_values["relu"] = self.relu.init_params(key)
        params["shortcut"], cached_values["shortcut"] = self.shortcut.init_params(
            key)
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        cached_values_ = {}
        o, cached_values_["direct"] = self.direct.apply(
            params["direct"], x, cached_values=cached_values["direct"])

        i, cached_values_["shortcut"] = self.shortcut.apply(params["shortcut"], x,
                                                            cached_values=cached_values["shortcut"])

        o = o + i
        o, cached_values_["relu"] = self.relu.apply(
            params["relu"], o, cached_values=cached_values["relu"])
        return o, cached_values_


class GRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

    def init_params(self, key):
        params = {}
        params["Wir"] = random.normal(key, (self.input_size, self.hidden_size))
        params["bir"] = random.normal(key, (self.hidden_size,))
        params["Whr"] = random.normal(
            key, (self.hidden_size, self.hidden_size))
        params["bhr"] = random.normal(key, (self.hidden_size,))
        params["Wiz"] = random.normal(key, (self.input_size, self.hidden_size))
        params["biz"] = random.normal(key, (self.hidden_size,))
        params["Whz"] = random.normal(
            key, (self.hidden_size, self.hidden_size))
        params["bhz"] = random.normal(key, (self.hidden_size,))
        params["Win"] = random.normal(key, (self.input_size, self.hidden_size))
        params["bin"] = random.normal(key, (self.hidden_size,))
        params["Whn"] = random.normal(
            key, (self.hidden_size, self.hidden_size))
        params["bhn"] = random.normal(key, (self.hidden_size,))

        return params, {}

    def apply(self, params, x, cached_values=None):
        def apply_fn(params, h, seq):
            Wir, bir, Whr, bhr = params["Wir"], params["bir"], params["Whr"], params["bhr"]
            Wiz, biz, Whz, bhz = params["Wiz"], params["biz"], params["Whz"], params["bhz"]
            Win, bin, Whn, bhn = params["Win"], params["bin"], params["Whn"], params["bhn"]
            r = sigmoid(seq @ Wir + bir + h @ Whr + bhr)
            z = sigmoid(seq @ Wiz + biz + h @ Whz + bhz)
            n = tanh(seq @ Win + bin + r * (h @ Whn + bhn))
            h = (1 - z) * n + z * h
            return h, h
        _, N, L = x.shape
        hidden = np.zeros((N, self.hidden_size))
        fun_f = partial(apply_fn, params)
        _, o = scan(fun_f, hidden, x)
        return o, cached_values
