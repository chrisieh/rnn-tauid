import argparse

import numpy as np

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(aspect_ratio=1.0, pad_bottom=0.12)

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
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.06)

    # ROC
    ax0 = plt.subplot(gs[0])
    ax0.plot(eff_ref, rej_ref, label="Reference")
    for eff, rej, score in zip(eff_add, rej_add, args.add_score):
        ax0.plot(eff, rej, label=score)

    ax0.set_yscale("log")
    ax0.set_ylim(1, 1e5)
    ax0.set_ylabel("Rejection", ha="right", y=1.0)
    ax0.tick_params(labelbottom="off")
    ax0.legend()

    # Ratio
    x = np.linspace(0.05, 1.0, 100)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, roc_interp_ref(x) / roc_interp_ref(x), label="Reference")

    for roc_interp, score in zip(roc_interp_add, args.add_score):
        ax1.plot(x, roc_interp(x) / roc_interp_ref(x), label=score)

    ax1.set_xlabel("Signal efficiency", ha="right", x=1.0)
    ax1.set_ylabel("Ratio")#, ha="right", y=1.0)

    ylim = ax1.get_ylim()
    ax1.set_ylim(min(*ylim) - 0.05, max(*ylim) + 0.05)

    fig.savefig(args.outfile)

    #e41a1c
    #377eb8
    #4daf4a
    #984ea3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigf")
    parser.add_argument("bkgf")
    parser.add_argument("ref_score")
    parser.add_argument("add_score", nargs="*")
    parser.add_argument("-o", dest="outfile", default="roc_comparison.pdf")

    args = parser.parse_args()
    main(args)