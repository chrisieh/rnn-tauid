from collections import namedtuple

import h5py
import numpy as np

from rnn_tauid.common.preprocessing import pt_reweight


Data = namedtuple("Data", ["x", "y", "w"])


def load_data(sig, bkg, sig_slice, bkg_slice, invars, num=None):
    # pt-reweighting
    sig_pt = sig["TauJets/pt"][sig_slice]
    bkg_pt = bkg["TauJets/pt"][bkg_slice]

    sig_weight, bkg_weight = pt_reweight(sig_pt, bkg_pt)
    w = np.concatenate([sig_weight, bkg_weight])

    sig_len = len(sig_pt)
    bkg_len = len(bkg_pt)

    del sig_pt, bkg_pt
    del sig_weight, bkg_weight

    # Class labels
    y = np.ones(sig_len + bkg_len, dtype=np.float32)
    y[sig_len:] = 0

    # Load variables
    n_vars = len(invars)

    # If number of timesteps given
    if num:
        x = np.empty((sig_len + bkg_len, num, n_vars))

        sig_src = np.s_[sig_slice, :num]
        bkg_src = np.s_[bkg_slice, :num]
    else:
        x = np.empty((sig_len + bkg_len, n_vars))

        sig_src = np.s_[sig_slice]
        bkg_src = np.s_[bkg_slice]

    for i, (varname, func, _) in enumerate(invars):
        sig_dest = np.s_[:sig_len, ..., i]
        bkg_dest = np.s_[sig_len:, ..., i]

        if func:
            func(sig, x, source_sel=sig_src, dest_sel=sig_dest)
            func(bkg, x, source_sel=bkg_src, dest_sel=bkg_dest)
        else:
            sig[varname].read_direct(x, source_sel=sig_src, dest_sel=sig_dest)
            bkg[varname].read_direct(x, source_sel=bkg_src, dest_sel=bkg_dest)

    return Data(x=x, y=y, w=w)


def parallel_shuffle(sequences):
    size = None
    for seq in sequences:
        if size:
            assert size == len(seq)
        size = len(seq)

        random_state = np.random.RandomState(seed=1234567890)
        random_state.shuffle(seq)


def train_test_split(data, test_size=0.2):
    if not isinstance(data, list):
        data = [data]

    assert len(data) >= 1

    train_size = 1.0 - test_size
    test_start, test_stop = int(train_size * len(data[0].y)), len(data[0].y)

    train = slice(0, test_start)
    test = slice(test_start, test_stop)

    arr = []
    for d in data:
        arr.extend([d.x, d.y, d.w])

    parallel_shuffle(arr)

    ret = []
    for d in data:
        ret.extend([Data(x=d.x[train], y=d.y[train], w=d.w[train]),
                    Data(x=d.x[test], y=d.y[test], w=d.w[test])])
    return ret


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


def load_data_pfo(data, data_slice, invars, num=None):
    # Class labels
    decaymode = data["TauJets/truthDecayMode"][data_slice]
    data_len = len(decaymode)
    w = np.ones(data_len, dtype=np.float32)

    y = (decaymode[:, np.newaxis] == np.unique(decaymode)).astype(np.float32)

    # Load variables
    n_vars = len(invars)

    # If number of timesteps given
    if num:
        x = np.empty((data_len, num, n_vars))
        data_src = np.s_[data_slice, :num]
    else:
        x = np.empty((data_len, n_vars))
        data_src = np.s_[data_slice]

    for i, (varname, func, _) in enumerate(invars):
        data_dest = np.s_[:data_len, ..., i]

        if func:
            func(data, x, source_sel=data_src, dest_sel=data_dest)
        else:
            data[varname].read_direct(x, source_sel=data_src, dest_sel=data_dest)

    return Data(x=x, y=y, w=w)
