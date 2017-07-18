import argparse

import numpy as np
import h5py
from scipy.stats import binned_statistic, ks_2samp

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from rnn_tauid.evaluation.misc import bin_center, bin_width


def plot_train(ax, s, sw, b, bw, bins=None):
    ax.hist(s, weights=sw, bins=bins, normed=True, histtype="step", ec="r",
            label="Signal (train)")
    ax.hist(b, weights=bw, bins=bins, normed=True, histtype="step", ec="b",
            label="Background (train)")
    return ax


def plot_test(ax, s, sw, b, bw, bins=None):
    sigh, _, _ = binned_statistic(s, sw, statistic="sum", bins=bins)
    bkgh, _, _ = binned_statistic(b, bw, statistic="sum", bins=bins)

    def poisson(x):
        return np.sqrt(np.sum(x ** 2))

    dsigh, _, _ = binned_statistic(s, sw, statistic=poisson, bins=bins)
    dbkgh, _, _ = binned_statistic(b, bw, statistic=poisson, bins=bins)

    x = bin_center(bins)
    dx = 0.5 * bin_width(bins)

    # Normalize
    sig_norm = np.sum(sigh * bin_width(bins))
    bkg_norm = np.sum(bkgh * bin_width(bins))

    sigh_norm = sigh / sig_norm
    bkgh_norm = bkgh / bkg_norm
    dsigh_norm = dsigh / sig_norm
    dbkgh_norm = dbkgh / bkg_norm

    ax.errorbar(x, sigh_norm, xerr=dx, yerr=dsigh_norm, c="r",
                fmt="o", label="Signal (test)")
    ax.errorbar(x, bkgh_norm, xerr=dx, yerr=dbkgh_norm, c="b",
                fmt="o", label="Background (test)")

    ax.set_xlabel("BDT score", ha="right", x=1.0)
    ax.set_ylabel("Normalised number of events", ha="right", y=1.0)

    return ax


def main(args):
    with h5py.File(args.file, "r") as f:
        train_tree = f["TrainTree"][...]
        test_tree = f["TestTree"][...]

    train_sig = (train_tree["classID"] == 0)
    test_sig = (test_tree["classID"] == 0)

    train_weight = train_tree["weight"]
    test_weight = test_tree["weight"]

    train_score = train_tree["classifier"]
    test_score = test_tree["classifier"]

    # KS-test
    sig_d, sig_pval = ks_2samp(train_score[train_sig], test_score[test_sig])
    bkg_d, bkg_pval = ks_2samp(train_score[~train_sig], test_score[~test_sig])

    bins = np.linspace(-1.0, 1.0, 30)

    fig, ax = plt.subplots()
    plot_train(ax,
               train_score[train_sig], train_weight[train_sig],
               train_score[~train_sig], train_weight[~train_sig],
               bins=bins)
    plot_test(ax,
              test_score[test_sig], test_weight[test_sig],
              test_score[~test_sig], test_weight[~test_sig],
              bins=bins)

    ax.set_ylim(args.y_range)
    leg = ax.legend(loc="upper center")
    ax.set_yscale("log")

    ax.text(0.3725, 0.52,
            "KS-Test $p$-values:\n$p_\\mathrm{{sig}} = {:.1f} \\, \\%$"
            "\n$p_\\mathrm{{bkg}} = {:.1f} \\, \\%$".format(100*sig_pval, 100*bkg_pval),
            va="top", transform=ax.transAxes)

    fig.savefig(args.outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-o", dest="outfile", required=True)
    parser.add_argument("--y-range", nargs=2, type=float, default=(0.05, 10))

    args = parser.parse_args()
    main(args)
