import argparse

import numpy as np
import h5py

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

import matplotlib.pyplot as plt


def main(args):
    with h5py.File(args.data, "r", driver="family", memb_size=10*1024**3) as f:
        truth = f["TauJets/truthDecayMode"]
        idx = int(0.5 * len(truth))
        truth = truth[idx:]

    with h5py.File(args.deco, "r") as f:
        pred = f["score"][idx:]

    labels = ["1p0n", "1p1n", "1pXn", "3p0n", "3pXn"]
    texlabels = [r"$h^\pm$", r"$h^\pm \pi^0$", r"$h^\pm \geq 2 \pi^0$",
                 r"$3 h^\pm$", r"$3 h^\pm \geq 1 \pi^0$"]
    proba = {}
    for i, label in enumerate(labels):
        sel = (truth == i)
        proba[label] = pred[sel]

    # Probability loop
    for i, (label1, texlabel1) in enumerate(zip(labels, texlabels)):
        fig, ax = plt.subplots()
        ax.set_xlabel("{} mode probability".format(texlabel1), ha="right", x=1)
        ax.set_ylabel("Normalised number of events", ha="right", y=1)

        # Hist constituent loop
        hists = {}
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        for j, (label2, texlabel2, c) in enumerate(zip(labels, texlabels,
                                                       colors)):
            if args.skip_3p and "3p" in label2:
                continue

            ax.hist(proba[label2][:, i], bins=40, range=(0, 1), normed=True,
                    histtype="step", color=c, label=texlabel2)
            ax.set_ylim(0, 10)
            ax.legend(loc="upper center", title="True decay mode:")

        fig.savefig("_".join([args.o, label1]) + ".pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("deco")
    parser.add_argument("-o", default="proba")

    parser.add_argument("--skip-3p", action="store_true")

    args = parser.parse_args()
    main(args)
