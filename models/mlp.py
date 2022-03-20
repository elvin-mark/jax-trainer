from utils import act_fun_dict, addmm
from jax import random


class MLP:
    def __init__(self, layers, act_fun="relu"):
        self.layers = layers
        self.act_fun = act_fun_dict[act_fun]

    def init_params(self, key):
        params = {}
        cached_values = None
        keyW, keyb = random.split(key)
        for k, (i, o) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            W = random.normal(keyW, (i, o)) / i ** 0.5
            b = random.normal(keyb, (o,)) / i ** 0.5
            params[str(k)] = {"W": W, "b": b}
        return params, cached_values

    def apply(self, params, x, cached_values=None):
        o = x
        for i in range(len(params) - 1):
            W, b = params[str(i)]["W"], params[str(i)]["b"]
            o = self.act_fun(addmm(o, W, b))
        last = len(params) - 1
        W, b = params[str(last)]["W"], params[str(last)]["b"]
        o = addmm(o, W, b)
        return o, cached_values
