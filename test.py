from jax import random
from models.conv import create_residual2d_classifier
from utils import train
from datasets import digits
from loss import CrossEntropyLoss
from optim import SGD

model = create_residual2d_classifier(image_shape=8)
key = random.PRNGKey(0)
params, cached_values = model.init_params(key)

loss_fn = CrossEntropyLoss(model)
optim = SGD(loss_fn, lr=0.01)

train_dl, test_dl = digits(flatten=False)

train(params, optim, train_dl, cached_values=cached_values)
