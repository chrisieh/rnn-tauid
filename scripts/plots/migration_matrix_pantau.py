import argparse
from itertools import product

import numpy as np
import h5py
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()
import matplotlib.pyplot as plt


def migration_matrix(truth, reco, comp=False):
    assert len(truth) == len(reco)
    diag_eff = np.count_nonzero(truth == reco) / float(len(truth))

    cm = confusion_matrix(truth, reco).T[::-1]

    # Normalize columns
    if comp:
        axis = 1
    else:
        axis = 0

    cm_norm = 100 * np.true_divide(cm, np.sum(cm, axis=axis, keepdims=True))

    return diag_eff, cm_norm


def main(args):
    with h5py.File(args.data, "r", driver="family", memb_size=10*1024**3) as f:
        nTracks = f["TauJets/nTracks"][...]
        mask = (nTracks == 1) | (nTracks == 3)

        truth = f["TauJets/truthDecayMode"][...]
        truth = truth[mask]
        pantau = f["TauJets/PanTau_DecayMode"][...]
        pantau = pantau[mask]

    diag_eff, cm = migration_matrix(truth, pantau, comp=args.composition)

    np.set_printoptions(suppress=True)
    print(cm)

    fig, ax = plt.subplots()
    ax.imshow(cm, vmin=0, vmax=100, cmap="Blues")

    ticklabels = ["1p0n", "1p1n", "1pXn", "3p0n", "3pXn"]

    ax.minorticks_off()
    ax.set_xticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(ticklabels)
    ax.get_xaxis().set_ticks_position("bottom")
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_yticklabels(ticklabels[::-1])
    ax.get_yaxis().set_ticks_position("left")

    ax.set_xlabel("True Tau Decay Mode", ha="right", x=1.0)
    ax.set_ylabel("Reco Tau Decay Mode", ha="right", y=1.0)

    # Set values on plot
    for i, j in product(range(cm.shape[0]), range(cm.shape[0])):
        ax.text(i, j, "{:.1f}".format(np.round(cm[j,i], decimals=1)+0), ha="center", va="center")

    ax.set_ylim(4.5, -1.5)
    ax.set_aspect(0.7)

    ax.text(4.25, -1, "Diagonal efficiency: {:.1f}%".format(100 * diag_eff),
            ha="right", va="center")

    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--composition", action="store_true")

    args = parser.parse_args()
    main(args)
