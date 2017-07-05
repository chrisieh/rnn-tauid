import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()


def main(args):
    df = pd.read_csv(args.results)

    xlim = 0, df.num.max() + 1

    maxval = (df.val_loss_mean + df.val_loss_std).max()
    ylim = (df.val_loss_mean - df.val_loss_std).min() - 0.02 * maxval, \
           1.02 * maxval

    fig, ax = plt.subplots()
    ax.errorbar(df.num, df.val_loss_mean, yerr=df.val_loss_std, fmt="o", c="r")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0, xlim[1] + 1, 2))
    ax.set_xticks(np.arange(1, xlim[1], 2), minor=True)

    xlabel = ""
    if args.track:
        xlabel = "Number of tracks"
    elif args.cluster:
        xlabel = "Number of clusters"

    ax.set_xlabel(xlabel, ha="right", x=1.0)
    ax.set_ylabel("Validation loss", ha="right", y=1.0)

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("-o", dest="outfile", default="nscan.pdf")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track", action="store_true")
    group.add_argument("--cluster", action="store_true")

    args = parser.parse_args()
    main(args)
