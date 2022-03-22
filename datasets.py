from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from utils import Dataloader, Dataset
import numpy as np
import os
import struct

data_root = "./data"
mnist_root = os.path.join(data_root, "MNIST")
lecun_mnist = "http://yann.lecun.com/exdb/mnist/"


def one_hot_encoding_(t, num_classes=10):
    tmp_ = np.zeros(10)
    tmp_[t] = 1
    return tmp_


def flatten_(x):
    return x.reshape((-1))


def digits(batch_size=32, flatten=True):
    data_ = load_digits()
    x_ = data_.images / 16.0
    N, H, W = x_.shape
    x_ = x_.reshape((N, 1, H, W))
    y_ = data_.target

    train_X, test_X, train_y, test_y = train_test_split(x_, y_, train_size=0.8)
    train_ds = Dataset(train_X, train_y, transform=flatten_ if flatten else None,
                       target_transform=one_hot_encoding_)
    test_ds = Dataset(test_X, test_y, transform=flatten_ if flatten else None,
                      target_transform=one_hot_encoding_)

    train_dl = Dataloader(train_ds, batch_size=batch_size)
    test_dl = Dataloader(test_ds, batch_size=batch_size)
    return train_dl, test_dl


def mnist(batch_size=32, flatten=True):
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    if not os.path.exists(mnist_root):
        os.mkdir(mnist_root)
    files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    for f in files:
        if not os.path.exists(os.path.join(mnist_root, f)):
            os.system(
                f"wget {lecun_mnist + f + '.gz'}; gunzip {f + '.gz'}; mv {f} {mnist_root}")

    tmp_f = open(os.path.join(mnist_root, files[0]), "rb").read()
    _, N, H, W = struct.unpack(">IIII", tmp_f[:16])
    train_x = np.array(struct.unpack(
        f">{N * H * W}B", tmp_f[16:])).reshape(N, 1, H, W) / 255.0
    if flatten:
        train_x = train_x.reshape(N, H * W)

    tmp_f = open(os.path.join(mnist_root, files[1]), "rb").read()
    _, N = struct.unpack(">II", tmp_f[:8])
    train_y = np.array(struct.unpack(f">{N}B", tmp_f[8:]))

    tmp_f = open(os.path.join(mnist_root, files[2]), "rb").read()
    _, N, H, W = struct.unpack(">IIII", tmp_f[:16])
    test_x = np.array(struct.unpack(
        f">{N * H * W}B", tmp_f[16:])).reshape(N, 1, H, W) / 255.0
    if flatten:
        test_x = test_x.reshape(N, H * W)

    tmp_f = open(os.path.join(mnist_root, files[3]), "rb").read()
    _, N = struct.unpack(">II", tmp_f[:8])
    test_y = np.array(struct.unpack(f">{N}B", tmp_f[8:]))

    train_ds = Dataset(train_x, train_y, transform=flatten_ if flatten else None,
                       target_transform=one_hot_encoding_)
    test_ds = Dataset(test_x, test_y, transform=flatten_ if flatten else None,
                      target_transform=one_hot_encoding_)

    train_dl = Dataloader(train_ds, batch_size=batch_size)
    test_dl = Dataloader(test_ds, batch_size=batch_size)

    return train_dl, test_dl
