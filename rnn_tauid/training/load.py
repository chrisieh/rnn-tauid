from collections import namedtuple

import h5py
import numpy as np


Data = namedtuple("Data", ["x", "y", "w"])


def load_data(filename, variables, num=None):
    with h5py.File(filename, "r") as f:
        ds = f["data"]

        label = f["label"][:]
        weight = f["weight"][:]

        if not num:
            _, num = ds[variables[0]].shape

        shape = len(label), num, len(variables)

        data = np.empty(shape, dtype=np.float32)
        for i, var in enumerate(variables):
            ds[var].read_direct(data, np.s_[...,:num], np.s_[..., i])

    return Data(x=data, y=label, w=weight)


def parallel_shuffle(sequences):
    size = None
    for seq in sequences:
        if size:
            assert size == len(seq)
        size = len(seq)

        random_state = np.random.RandomState(seed=1234567890)
        random_state.shuffle(seq)


def train_test_split(data, test_size=0.2):
    train_size = 1.0 - test_size
    test_start, test_stop = int(train_size * len(data.y)), len(data.y)

    train = slice(0, test_start)
    test = slice(test_start, test_stop)

    parallel_shuffle([data.x, data.y, data.w])

    return Data(x=data.x[train], y=data.y[train], w=data.w[train]), \
           Data(x=data.x[test], y=data.y[test], w=data.w[test])


def preprocess(train, test, funcs):
    preprocessing = []
    for i, func in enumerate(funcs):
        xi_train = train.x[..., i]
        xi_test = test.x[..., i]

        # Scale & offset from train, apply to test
        if func:
            offset, scale = func(xi_train)
            xi_train -= offset
            xi_train /= scale

            xi_test -= offset
            xi_test /= scale
        else:
            num = xi_train.shape[1]
            offset = np.zeros((num,), dtype=np.float32)
            scale = np.ones((num,), dtype=np.float32)

        preprocessing.append((offset, scale))

        # Replace nan with zero
        xi_train[np.isnan(xi_train)] = 0
        xi_test[np.isnan(xi_test)] = 0

    return preprocessing


def save_preprocessing(filename, variables, preprocessing):
    with h5py.File(filename, "w") as f:
        # Save variable names
        f["variables"] = np.array(variables, "S")

        # Save preprocessing
        for var, (offset, scale) in zip(variables, preprocessing):
            f[var + "/offset"] = offset
            f[var + "/scale"] = scale
