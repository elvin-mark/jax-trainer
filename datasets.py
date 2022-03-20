from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from utils import Dataloader, Dataset
import numpy as np


def digits(batch_size=32, flatten=True):
    data_ = load_digits()
    x_ = data_.data / 16.0
    y_ = data_.target

    def target_transform(t):
        tmp_ = np.zeros(10)
        tmp_[t] = 1
        return tmp_

    if flatten:
        transform = None
    else:
        def transform(x): return x.reshape((1, 1, 8, 8))

    train_X, test_X, train_y, test_y = train_test_split(x_, y_, train_size=0.8)
    train_ds = Dataset(train_X, train_y, transform=transform,
                       target_transform=target_transform)
    test_ds = Dataset(test_X, test_y, transform=transform,
                      target_transform=target_transform)

    train_dl = Dataloader(train_ds, batch_size=batch_size)
    test_dl = Dataloader(test_ds, batch_size=batch_size)
    return train_dl, test_dl
