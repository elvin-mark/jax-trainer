from utils import mse_loss, nll_loss, logsoftmax


class MSELoss:
    def __init__(self, model):
        self.model = model

    def __call__(self, params, x, t, cached_values=None):
        o, cached_values = self.model.apply(
            params, x, cached_values=cached_values)
        return mse_loss(o, t), cached_values


class NLLLoss:
    def __init__(self, model):
        self.model = model

    def __call__(self, params, x, t, cached_values=None):
        o, cached_values = self.model.apply(
            params, x, cached_values=cached_values)
        return nll_loss(o, t), cached_values


class CrossEntropyLoss:
    def __init__(self, model):
        self.model = model

    def __call__(self, params, x, t, cached_values=None):
        o, cached_values = self.model.apply(
            params, x, cached_values=cached_values)
        o = logsoftmax(o)
        return nll_loss(o, t), cached_values
