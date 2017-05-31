import argparse

import numpy as np
import h5py

import matplotlib as mpl
mpl.use("PDF")

from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main(args):
    with h5py.File(args.data, "r", driver="family", memb_size=10*1024**3) as f:
        truth = f["TauJets/truthDecayMode"]
        idx = int(0.5 * len(truth))
        truth = truth[idx:]

    with h5py.File(args.deco, "r") as f:
        pred = f["score"][idx:]

    labels = ["1p0n", "1p1n", "1pXn", "3p0n", "3pXn"]
    proba = {}
    for i, label in enumerate(labels):
        sel = (truth == i)
        proba[label] = pred[sel]


    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 2)

    # Probability loop
    for i, label1 in enumerate(labels):
        ax = plt.subplot(gs[i // 2, i % 2])
        ax.set_xlabel("{} probability".format(label1), ha="right", x=1.0)

        # Hist constituent loop
        hists = {}
        for j, label2 in enumerate(labels):
            ax.hist(proba[label2][:, i], bins=40, range=(0, 1), normed=True,
                                    histtype="step", label=label2)
            ax.set_ylim(0, 10)

        if i == 1:
            ax.legend()

    gs.tight_layout(fig, pad=0.4)
    fig.savefig(args.o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("deco")
    parser.add_argument("-o", default="proba.pdf")

    args = parser.parse_args()
    main(args)
