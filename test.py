from jax import random
from matplotlib.cbook import flatten
from models.conv import Conv2dNN
from utils import train
from datasets import digits
from loss import CrossEntropyLoss
from optim import SGD

feature_extractor = [
    {
        "in_planes": 1,
        "out_planes": 16,
        "kernel_size": 3,
        "max_pool": False
    },
    {
        "in_planes": 16,
        "out_planes": 32,
        "kernel_size": 3,
        "max_pool": False
    },
    {
        "in_planes": 32,
        "out_planes": 64,
        "kernel_size": 3,
        "max_pool": False
    }
]

num_features = 256
num_classes = 10

model = Conv2dNN(feature_extractor, num_features, num_classes)
key = random.PRNGKey(0)
params, cached_values = model.init_params(key)

loss_fn = CrossEntropyLoss(model)
optim = SGD(loss_fn, lr=0.01)

train_dl, test_dl = digits(flatten=False)

train(params, optim, train_dl, cached_values=cached_values)
