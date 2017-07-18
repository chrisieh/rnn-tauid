import argparse

import numpy as np

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, aspect_ratio=1.0, pad_bottom=0.12)

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def roc(*args, **kwargs):
    fpr, tpr, thr = roc_curve(*args, **kwargs)
    nonzero = fpr != 0
    eff = tpr[nonzero]
    rej = 1.0 / fpr[nonzero]

    return eff, rej


def main(args):
    from root_numpy import root2array

    tree = "CollectionTree"
    weight = "weight"

    # Reference BDT
    ref_sig_data = root2array(args.sigf, treename=tree,
                              branches=[args.ref_score, weight])
    ref_bkg_data = root2array(args.bkgf, treename=tree,
                              branches=[args.ref_score, weight])

    # Additional BDTs
    add_sig_data = []
    add_bkg_data = []
    for score in args.add_score:
        add_sig_data.append(root2array(args.sigf, treename=tree,
                                       branches=score))
        add_bkg_data.append(root2array(args.bkgf, treename=tree,
                                       branches=score))

    # Weights and true labels are the same for all BDTs
    w = np.concatenate([ref_sig_data[weight], ref_bkg_data[weight]])
    y_true = np.concatenate([np.ones(len(ref_sig_data)),
                             np.zeros(len(ref_bkg_data))])

    # Get predicted values
    y_ref = np.concatenate([ref_sig_data[args.ref_score],
                            ref_bkg_data[args.ref_score]])
    y_add = []
    for sig, bkg in zip(add_sig_data, add_bkg_data):
        y_add.append(np.concatenate([sig, bkg]))

    # Calculates roc
    eff_ref, rej_ref = roc(y_true, y_ref, sample_weight=w)
    roc_interp_ref = interp1d(eff_ref, rej_ref)

    # Calculates roc for add. BDTs
    eff_add = []
    rej_add = []
    roc_interp_add = []
    for y in y_add:
        eff, rej = roc(y_true, y, sample_weight=w)
        eff_add.append(eff)
        rej_add.append(rej)
        roc_interp_add.append(interp1d(eff, rej))

    # Plotting code
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

    # ROC
    ax0 = plt.subplot(gs[0])
    ax0.plot(eff_ref, rej_ref, c="k", label="Reference")
    for eff, rej, score, c, l in zip(eff_add, rej_add, args.add_score,
                                     ["r", "b"], ["A", "B"]):
        label = "BDT {}".format(l)
        ax0.plot(eff, rej, c=c, label=label)

    ax0.set_yscale("log")
    if not args.mode3p:
        ax0.set_ylim(1, 1e4)
    else:
        ax0.set_ylim(1, 1e4)
    ax0.set_ylabel("Rejection", ha="right", y=1.0)
    ax0.tick_params(labelbottom="off")
    ax0.legend()

    # Ratio
    xfull = np.linspace(0.0, 1.0, 10)
    if not args.mode3p:
        x = np.linspace(0.1, 1.0, 200)
    else:
        x = np.linspace(0.2, 1.0, 200)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(xfull, np.ones_like(xfull), c="k", label="Reference")

    for roc_interp, score, c in zip(roc_interp_add, args.add_score, ["r", "b"]):
        ax1.plot(x, roc_interp(x) / roc_interp_ref(x), c=c)

    ax1.set_xlabel("Signal efficiency", ha="right", x=1.0)
    ax1.set_ylabel("Ratio")

    ylim = ax1.get_ylim()
    if not args.mode3p:
        ax1.set_ylim(0.9, 1.5)
        ax1.set_yticks([1.0, 1.2, 1.4])
    else:
        ax1.set_ylim(0.95, 1.3)
        ax1.set_yticks([1.0, 1.1, 1.2])

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigf")
    parser.add_argument("bkgf")
    parser.add_argument("ref_score")
    parser.add_argument("add_score", nargs="*")
    parser.add_argument("-o", dest="outfile", default="roc_comparison.pdf")
    parser.add_argument("--mode3p", action="store_true")

    args = parser.parse_args()
    main(args)
