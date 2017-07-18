import argparse

import numpy as np


import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18, pad_right=0.91)

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width
from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


def main(args):
    from root_numpy import root2array

    # Load data
    score = args.score
    treename = "CollectionTree"
    branches = ["TauJets.pt", "TauJets.mu"] + [score]

    sig = root2array(args.sig, treename=treename, branches=branches)

    # Calculate working point
    flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
    sig_pass_thr = flat.fit(sig["TauJets.pt"], sig["TauJets.mu"], sig[score])

    # Check if flattening meets efficiency goal
    assert np.isclose(np.count_nonzero(sig_pass_thr) / float(len(sig_pass_thr)),
                      args.eff, atol=0, rtol=1e-2)

    # Plot working point
    fig, ax = plt.subplots()

    # Subtract 0.5 to make mu bins integers instead of XX.5's
    xx, yy = np.meshgrid(flat.x_bins, flat.y_bins - 0.5)
    cm = ax.pcolormesh(xx / 1000.0, yy, flat.cutmap.T)

    ax.set_xscale("log")
    ax.set_xlim(20, 2000)
    ax.set_xticks([20, 1e2, 1e3])
    ax.set_xticklabels(["$20$", "$100$", "$1000$"])
    ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV",
                  ha="right", x=1)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Average interactions\nper bunch crossing $\mu$",
                  ha="right", y=1)

    cb = fig.colorbar(cm)
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=4.0)
    cb.set_label("BDT score cut", ha="right", y=1.0)

    if args.label:
        ax.text(0.93, 0.07, args.label, ha="right", va="bottom",
                fontsize=7, transform=ax.transAxes)

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("score")

    parser.add_argument("--eff", type=float, default=0.6)
    parser.add_argument("-o", dest="outfile", default="wp.pdf")
    parser.add_argument("--label", default=None)

    args = parser.parse_args()
    main(args)
