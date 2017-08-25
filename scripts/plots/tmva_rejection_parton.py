import argparse

import numpy as np
from tqdm import tqdm

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18, pad_bottom=0.18)

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
    bkg = root2array(args.bkg, treename=treename,
                     branches=branches + ["TauJets.PartonTruthLabelID"])

    g_mask = bkg["TauJets.PartonTruthLabelID"] == 21
    q_mask = (bkg["TauJets.PartonTruthLabelID"] >= 0) & \
             (bkg["TauJets.PartonTruthLabelID"] <  5)
    b_mask = bkg["TauJets.PartonTruthLabelID"] == 5
    #c_mask = bkg["TauJets.PartonTruthLabelID"] == 4

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

    flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
    sig_pass_thr = flat.fit(sig["TauJets.pt"], sig["TauJets.mu"],
                            sig[score])

    # Check if flattening meets efficiency goal
    assert np.isclose(np.count_nonzero(sig_pass_thr) / float(len(sig_pass_thr)),
                      args.eff, atol=0, rtol=1e-2)

    bkg_pass_thr = flat.passes_thr(bkg["TauJets.pt"], bkg["TauJets.mu"],
                                   bkg[score])


    # Quarks
    q_bkg_eff = binned_efficiency(bkg["TauJets.pt"][q_mask], bkg_pass_thr[q_mask],
                                    bins=bins)
    q_bkg_rej = 1.0 / q_bkg_eff.mean
    q_d_bkg_rej = q_bkg_eff.std / q_bkg_eff.mean ** 2

    q_rej = q_bkg_rej
    q_d_rej = q_d_bkg_rej


    # Gluons
    g_bkg_eff = binned_efficiency(bkg["TauJets.pt"][g_mask], bkg_pass_thr[g_mask],
                                  bins=bins)
    g_bkg_rej = 1.0 / g_bkg_eff.mean
    g_d_bkg_rej = g_bkg_eff.std / g_bkg_eff.mean ** 2

    g_rej = g_bkg_rej
    g_d_rej = g_d_bkg_rej

    # b-jets
    b_bkg_eff = binned_efficiency(bkg["TauJets.pt"][b_mask], bkg_pass_thr[b_mask],
                                  bins=bins)
    b_bkg_rej = 1.0 / b_bkg_eff.mean
    b_d_bkg_rej = b_bkg_eff.std / b_bkg_eff.mean ** 2

    b_rej = b_bkg_rej
    b_d_rej = b_d_bkg_rej

    # c-jets
    # c_bkg_eff = binned_efficiency(bkg["TauJets.pt"][c_mask], bkg_pass_thr[c_mask],
    #                               bins=bins)
    # c_bkg_rej = 1.0 / c_bkg_eff.mean
    # c_d_bkg_rej = c_bkg_eff.std / c_bkg_eff.mean ** 2

    # c_rej = c_bkg_rej
    # c_d_rej = c_d_bkg_rej


    # Plotting
    fig, ax = plt.subplots()
    ax.errorbar(bin_midpoint / 1000.0, q_rej,
                xerr=bin_half_width / 1000.0, yerr=q_d_rej,
                fmt="o", c="r", zorder=-1, label="quark jets (udsc)")
    ax.errorbar(bin_midpoint / 1000.0, b_rej,
                xerr=bin_half_width / 1000.0, yerr=b_d_rej,
                fmt="o", c="k", zorder=-2, label="b-jets")
    # ax.errorbar(bin_midpoint / 1000.0, c_rej,
    #             xerr=bin_half_width / 1000.0, yerr=c_d_rej,
    #             fmt="o", c="c", zorder=-2, label="c-jets")

    if not args.div_gluons:
        ax.errorbar(bin_midpoint / 1000.0, g_rej,
                    xerr=bin_half_width / 1000.0, yerr=g_d_rej,
                    fmt="o", c="b", zorder=-3, label="gluon jets")
    else:
        ax.errorbar(bin_midpoint / 1000.0, g_rej / 2.0,
                    xerr=bin_half_width / 1000.0, yerr=g_d_rej / 2.0,
                    fmt="o", c="b", zorder=-3, label="gluon jets (Rej. / 2)")

    ax.set_xlim(20, args.pt_max)
    ax.set_ylabel("Rejection", ha="right", y=1.0)
    ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)

    if args.ylim:
        ax.set_ylim(*args.ylim)

    if args.yticks_ratio:
        ax1.set_yticks(args.yticks_ratio)

    if args.y_zero:
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])

    ax.text(0.06, 0.94,
            "{}% signal efficiency working point".format(int(100 * args.eff)),
            va="top", fontsize=7, transform=ax.transAxes)
    ax.legend(bbox_to_anchor=(0.0, 0.88), bbox_transform=ax.transAxes,
              loc="upper left", fontsize=7)
    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("score")

    parser.add_argument("--eff", type=float, default=0.6)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--quantiles", action="store_true")
    parser.add_argument("--pt-max", type=float, default=200)
    parser.add_argument("--y-zero", action="store_true")
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--yticks-ratio", nargs="+", type=float, default=None)
    parser.add_argument("-o", dest="outfile", default="rej.pdf")
    parser.add_argument("--div-gluons", action="store_true")

    args = parser.parse_args()
    main(args)
