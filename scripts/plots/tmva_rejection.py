import argparse

import numpy as np
from tqdm import tqdm

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18, pad_bottom=0.12, aspect_ratio=1.0)

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

    bin_midpoint = bin_center(bins)
    bin_half_width = bin_width(bins) / 2.0

    rej = {}
    d_rej = {}
    flatteners = {}

    for score in scores:
        # Calculate working point
        flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
        sig_pass_thr = flat.fit(sig["TauJets.pt"], sig["TauJets.mu"], sig[score])
        flatteners[score] = flat

        # Check if flattening meets efficiency goal
        assert np.isclose(np.count_nonzero(sig_pass_thr) / float(len(sig_pass_thr)),
                          args.eff, atol=0, rtol=1e-2)

        # Background events passing working point
        bkg_pass_thr = flat.passes_thr(bkg["TauJets.pt"], bkg["TauJets.mu"],
                                       bkg[score])

        # Background efficiency & rejection
        bkg_eff = binned_efficiency(bkg["TauJets.pt"], bkg_pass_thr, bins=bins)
        bkg_rej = 1.0 / bkg_eff.mean
        d_bkg_rej = bkg_eff.std / bkg_eff.mean ** 2

        rej[score] = bkg_rej
        d_rej[score] = d_bkg_rej


    ratio = {}
    # Bootstrap the ratio
    n_bootstrap = args.n_bootstrap
    for score in tqdm(scores[1:]):
        ratio[score] = []

        for i in tqdm(range(n_bootstrap)):
            idx = np.random.randint(len(bkg), size=len(bkg))
            bootstrap = bkg[idx]

            # Reference
            flat_ref = flatteners[scores[0]]
            bkg_pass_thr_ref = flat_ref.passes_thr(bootstrap["TauJets.pt"],
                                                   bootstrap["TauJets.mu"],
                                                   bootstrap[scores[0]])
            bkg_eff_ref = binned_efficiency(bootstrap["TauJets.pt"],
                                            bkg_pass_thr_ref,
                                            bins=bins)
            bkg_rej_ref = 1.0 / bkg_eff_ref.mean

            # To compare
            flat = flatteners[score]
            bkg_pass_thr = flat.passes_thr(bootstrap["TauJets.pt"],
                                           bootstrap["TauJets.mu"],
                                           bootstrap[score])

            # Background efficiency & rejection
            bkg_eff = binned_efficiency(bootstrap["TauJets.pt"], bkg_pass_thr,
                                        bins=bins)
            bkg_rej = 1.0 / bkg_eff.mean

            ratio[score].append(bkg_rej / bkg_rej_ref)

    ratio_mean = {}
    ratio_std = {}
    for key in ratio:
        r = np.array(ratio[key])
        ratio_mean[key] = r.mean(axis=0)
        ratio_std[key] = r.std(axis=0)

    # Plotting
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

    ax = plt.subplot(gs[0])
    for score, label, color, zorder in zip(scores,
                                           args.labels,
                                           ["k", "r", "b"], [-3, -2, -1]):
        ax.errorbar(bin_midpoint / 1000.0, rej[score],
                    xerr=bin_half_width / 1000.0, yerr=d_rej[score],
                    fmt="o", c=color, zorder=zorder, label=label)

    ax.set_xlim(20, args.pt_max)
    ax.set_ylabel("Rejection", ha="right", y=1.0)

    # Ratio plot
    ax1 = plt.subplot(gs[1], sharex=ax)
    ref_score, cmp_scores = scores[0], scores[1:]

    for score, color, zorder in zip(cmp_scores, ["r", "b"], [-2, -1]):
        ax1.errorbar(bin_midpoint / 1000.0, ratio_mean[score],
                     xerr=bin_half_width / 1000.0, yerr=ratio_std[score],
                     fmt="o", c=color, zorder=zorder)

    ax1.set_ylabel("Ratio")
    ax.tick_params(labelbottom="off")
    ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)

    if args.ylim:
        ax.set_ylim(*args.ylim)

    if args.ylim_ratio:
        ax1.set_ylim(*args.ylim_ratio)

    if args.yticks_ratio:
        ax1.set_yticks(args.yticks_ratio)

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
    parser.add_argument("--ylim-ratio", nargs=2, type=float, default=None)
    parser.add_argument("--xticks", nargs="+", type=float, default=None)
    parser.add_argument("--yticks-ratio", nargs="+", type=float, default=None)
    parser.add_argument("-o", dest="outfile", default="rej.pdf")
    parser.add_argument("--labels", nargs="+", default=["Reference",
                                                        "BDT A", "BDT B"])
    parser.add_argument("--n-bootstrap", type=int, default=50)

    args = parser.parse_args()
    main(args)
