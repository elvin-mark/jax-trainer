from utils import conv2d, max_pool2d, batch_norm2d, relu
from jax import random
import numpy as np


class Conv2dNN:
    def __init__(self, feature_extractor, num_features, num_classes):
        self.feature_extractor = feature_extractor
        self.num_features = num_features
        self.num_classes = num_classes

    def init_params(self, key):
        params = {}
        cached_values = {}
        for i, tmp in enumerate(self.feature_extractor):
            layer_ = f"conv_block_{i}"
            params[layer_] = {}
            params[layer_]["W"] = random.normal(
                key, (tmp["out_planes"], tmp["in_planes"], tmp["kernel_size"], tmp["kernel_size"]))
            params[layer_]["gamma"] = random.normal(
                key, (1, tmp["out_planes"], 1, 1))
            params[layer_]["beta"] = random.normal(
                key, (1, tmp["out_planes"], 1, 1))
            cached_values[layer_] = {}
            cached_values[layer_]["running_mean"] = np.zeros(
                (1, tmp["out_planes"], 1, 1))
            cached_values[layer_]["running_var"] = np.ones(
                (1, tmp["out_planes"], 1, 1))

        params["fc"] = {}
        params["fc"]["W"] = random.normal(
            key, (self.num_features, self.num_classes))
        params["fc"]["b"] = random.normal(
            key, (self.num_classes,))

        return params, cached_values

    def apply(self, params, x, cached_values=None):
        o = x
        for i, tmp in enumerate(self.feature_extractor):
            layer_ = f"conv_block_{i}"
            params_ = params[layer_]
            W, gamma, beta = params_["W"], params_["gamma"], params_["beta"]
            running_mean, running_var = cached_values[layer_][
                "running_mean"], cached_values[layer_]["running_var"]
            o = conv2d(o, W)
            o, running_mean, running_var = batch_norm2d(
                o, gamma, beta, running_mean, running_var)
            cached_values[layer_]["running_mean"] = running_mean
            cached_values[layer_]["running_var"] = running_var
            o = relu(o)
            if tmp["max_pool"]:
                o = max_pool2d(o)

        o = o.reshape(-1, self.num_features)

        W, b = params["fc"]["W"], params["fc"]["b"]
        o = o @ W + b
        return o, cached_values
