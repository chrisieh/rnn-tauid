import argparse

import numpy as np

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18)

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width
from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


def main(args):
    from root_numpy import root2array

    # Load data
    scores = args.scores
    treename = "CollectionTree"
    branches = ["TauJets.pt", "TauJets.mu"] + scores

    sig = root2array(args.sig, treename=treename, branches=branches)
    bkg = root2array(args.bkg, treename=treename, branches=branches)

    if not args.quantiles:
        bins = 10 ** np.linspace(np.log10(20000), np.log10(args.pt_max * 1000.0),
                                 args.bins + 1)
    else:
        bins = np.percentile(
            sig["TauJets.pt"][sig["TauJets.pt"] < 1000 * args.pt_max],
            np.linspace(0, 100, args.bins + 1))

    rej = {}
    d_rej = {}

    for score in scores:
        # Calculate working point
        flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
        sig_pass_thr = flat.fit(sig["TauJets.pt"], sig["TauJets.mu"], sig[score])

        # Check if flattening meets efficiency goal
        assert np.isclose(np.count_nonzero(sig_pass_thr) / float(len(sig_pass_thr)),
                          args.eff, atol=0, rtol=1e-2)

        # Background events passing working point
        bkg_pass_thr = flat.passes_thr(bkg["TauJets.pt"], bkg["TauJets.mu"],
                                       bkg[score])
        bin_midpoint = bin_center(bins)
        bin_half_width = bin_width(bins) / 2.0

        # Background efficiency & rejection
        bkg_eff = binned_efficiency(bkg["TauJets.pt"], bkg_pass_thr, bins=bins)
        bkg_rej = 1.0 / bkg_eff.mean
        d_bkg_rej = bkg_eff.std / bkg_eff.mean ** 2

        rej[score] = bkg_rej
        d_rej[score] = d_bkg_rej

    # Plotting
    fig, ax = plt.subplots()

    for score, label, color, zorder in zip(scores,
                                           ["Reference", "BDT A", "BDT B"],
                                           ["k", "r", "b"], [-3, -2, -1]):
        ax.errorbar(bin_midpoint / 1000.0, rej[score],
                    xerr=bin_half_width / 1000.0, yerr=d_rej[score],
                    fmt="o", c=color, zorder=zorder, label=label)

    ax.set_xlim(20, args.pt_max)
    ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)
    ax.set_ylabel("Rejection", ha="right", y=1.0)


    # Prepend 20 GeV tick if it does not exist
    # xticks = ax.get_xticks()
    # if not xticks[0] == 20.0:
    #     ax.set_xticks([20] + list(xticks))

    if args.ylim:
        ax.set_ylim(*args.ylim)

    if args.y_zero:
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])

    ax.text(0.06, 0.94,
            "{}% signal efficiency working point".format(int(100 * args.eff)),
            va="top", fontsize=7, transform=ax.transAxes)

    ax.legend(loc="lower right")

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("scores", nargs="+")

    parser.add_argument("--eff", type=float, default=0.6)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--quantiles", action="store_true")
    parser.add_argument("--pt-max", type=float, default=300)
    parser.add_argument("--y-zero", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--xticks", nargs="+", type=float, default=None)
    parser.add_argument("-o", dest="outfile", default="rej.pdf")

    args = parser.parse_args()
    main(args)
