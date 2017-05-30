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


def migration_matrix(truth, reco):
    assert len(truth) == len(reco)
    diag_eff = np.count_nonzero(truth == reco) / float(len(truth))

    cm = confusion_matrix(truth, reco).T[::-1]

    # Normalize columns
    cm_norm = 100 * np.true_divide(cm, np.sum(cm, axis=0, keepdims=True))

    return diag_eff, cm_norm


def main(args):
    with h5py.File(args.data, "r", driver="family", memb_size=10*1024**3) as f:
        nTracks = f["TauJets/nTracks"][...]
        mask = (nTracks == 1) | (nTracks == 3)

        truth = f["TauJets/truthDecayMode"][...]

        # Mask out training set
        idx = int(0.5 * len(truth))
        mask[:idx] = False

        truth = truth[mask]

    with h5py.File(args.deco, "r") as f:
        reco = f["score"][...]
        pass_margin = np.subtract.reduce(-np.partition(-reco, 2)[:, :2], axis=-1) > args.margin
        pass_margin = pass_margin[mask]
        reco = np.argmax(reco, axis=1)
        reco = reco[mask]

    # Modes
    mode_1p0n = truth == 0
    mode_1p1n = truth == 1
    mode_1pXn = truth == 2
    mode_3p0n = truth == 3
    mode_3pXn = truth == 4

    # Efficiencies
    eff_1p0n = np.count_nonzero(pass_margin[mode_1p0n]) / float(len(pass_margin[mode_1p0n]))
    eff_1p1n = np.count_nonzero(pass_margin[mode_1p1n]) / float(len(pass_margin[mode_1p1n]))
    eff_1pXn = np.count_nonzero(pass_margin[mode_1pXn]) / float(len(pass_margin[mode_1pXn]))
    eff_3p0n = np.count_nonzero(pass_margin[mode_3p0n]) / float(len(pass_margin[mode_3p0n]))
    eff_3pXn = np.count_nonzero(pass_margin[mode_3pXn]) / float(len(pass_margin[mode_3pXn]))

    print("Eff. 1p0n: {}".format(eff_1p0n))
    print("Eff. 1p1n: {}".format(eff_1p1n))
    print("Eff. 1pXn: {}".format(eff_1pXn))
    print("Eff. 3p0n: {}".format(eff_3p0n))
    print("Eff. 3pXn: {}".format(eff_3pXn))

    diag_eff, cm = migration_matrix(truth[pass_margin], reco[pass_margin])

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

    ax.text(3, -1.3, "Diag eff.: {:.1f}%".format(100 * diag_eff), va="center")

    # Mode efficiencies
    ax.text(-0.4, -1.3, "1p0n eff.: {:.1f}%".format(100 * eff_1p0n), va="center")
    ax.text(-0.4, -1.0, "1p1n eff.: {:.1f}%".format(100 * eff_1p1n), va="center")
    ax.text(-0.4, -0.7, "1pXn eff.: {:.1f}%".format(100 * eff_1pXn), va="center")
    ax.text(1.3, -1.0, "3p0n eff.: {:.1f}%".format(100 * eff_3p0n), va="center")
    ax.text(1.3, -0.7, "3pXn eff.: {:.1f}%".format(100 * eff_3pXn), va="center")


    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("deco")
    parser.add_argument("out")
    parser.add_argument("--margin", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
