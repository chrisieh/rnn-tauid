import argparse

import numpy as np
from root_numpy import root2array
from scipy.stats import binned_statistic

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18, pad_bottom=0.24)

from rnn_tauid.evaluation.misc import bin_center, bin_width
from rnn_tauid.common.preprocessing import pt_reweight


def overlay(s, sw, b, bw, bins=None, skip=False):
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

    fig, ax = plt.subplots()
    ax.errorbar(x, sigh_norm, xerr=dx, yerr=dsigh_norm, c="r",
                fmt="o", label="Signal")

    if not skip:
        ax.errorbar(x, bkgh_norm, xerr=dx, yerr=dbkgh_norm, c="b",
                    fmt="o", label="Background")

    ylim = 1.05 * np.max(np.concatenate([sigh_norm, bkgh_norm]))
    ax.set_ylim(0, ylim)

    if not skip:
        ax.legend()

    return fig, ax


def main(args):
    sig = root2array(args.sig, treename="CollectionTree",
                     branches=["TauJets.pt", "TauJets.mu"])
    bkg = root2array(args.bkg, treename="CollectionTree",
                     branches=["TauJets.pt", "TauJets.mu"])

    sig_pt = sig["TauJets.pt"] / 1000.0
    bkg_pt = bkg["TauJets.pt"] / 1000.0
    sig_mu = sig["TauJets.mu"]
    bkg_mu = bkg["TauJets.mu"]

    pt_bins = np.linspace(20.0, 2000.0, 30)
    mu_bins = np.arange(0, 61, 2)

    fig, ax = overlay(sig_pt, np.ones_like(sig_pt),
                      bkg_pt, np.ones_like(bkg_pt),
                      bins=pt_bins)
    ax.set_xlabel("Reconstructed tau $p_\mathrm{T}$", ha="right", x=1.0)
    ax.set_ylabel("Normalised number of events", ha="right", y=1.0)
    ax.set_ylim(1e-5, 2e-2)
    ax.set_yscale("log")
    fig.savefig("pt.pdf")

    fig2, ax2 = overlay(sig_mu, np.ones_like(sig_mu),
                        bkg_mu, np.ones_like(bkg_mu),
                        bins=mu_bins, skip=True)
    ax2.set_xlabel("Average interactions\nper bunch crossing $\mu$", ha="right", x=1.0)
    ax2.set_ylabel("Normalised number of events", ha="right", y=1.0)
    ax2.set_ylim(0.0, 0.04)
    ax2.set_xlim(0, 60)
    fig2.savefig("mu.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")

    args = parser.parse_args()
    main(args)
