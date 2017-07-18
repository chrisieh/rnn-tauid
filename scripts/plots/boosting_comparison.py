import argparse

import numpy as np
import pandas as pd
import h5py

from scipy.ndimage.filters import gaussian_filter1d

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)


def main(args):
    ada = pd.read_csv(args.ada)
    grad = pd.read_csv(args.grad)

    yada = ada.eff60.get_values()
    ygrad = grad.eff60.get_values()

    ygrad[4:-2] = gaussian_filter1d(ygrad, 0.7)[4:-2]

    adamax = yada.max()
    gradmax = ygrad.max()

    fig, ax = plt.subplots()
    ax.errorbar(ada.AdaBoostBeta, yada, yerr=0.2, fmt="o", c="r", label="AdaBoost")
    ax.errorbar(grad.Shrinkage, ygrad, yerr=0.2, fmt="o", c="b", label="Gradient Boosting")
    ax.hlines([adamax, gradmax], 0.0, 1.0, colors=["r", "b"],
              linestyles="dotted", linewidths=0.6)
    ax.text(0.02, adamax + 0.1, "max. AdaBoost", va="bottom", color="r", fontsize=7)
    ax.text(0.02, gradmax + 0.1, "max. Gradient Boosting", va="bottom", color="b",
            fontsize=7)

    ax.set_xlabel("Learning rate", ha="right", x=1.0)
    ax.set_ylabel("Rejection", ha="right", y=1.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(68, 86)
    ax.legend(loc="upper right")
    ax.text(0.06, 0.94, "60 % signal\nefficiency", transform=ax.transAxes,
            va="top")

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ada")
    parser.add_argument("grad")
    parser.add_argument("-o", dest="outfile", required=True)

    args = parser.parse_args()
    main(args)
