import argparse

import numpy as np
import h5py
from scipy.stats import binned_statistic

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from rnn_tauid.evaluation.misc import bin_center, bin_width
from rnn_tauid.common.preprocessing import pt_reweight


def overlay(s, sw, b, bw, bins=None):
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
    ax.errorbar(x, bkgh_norm, xerr=dx, yerr=dbkgh_norm, c="b",
                fmt="o", label="Background")

    ylim = 1.05 * np.max(np.concatenate([sigh_norm, bkgh_norm]))
    ax.set_ylim(0, ylim)

    ax.legend()

    return fig, ax


def main(args):
    with h5py.File(args.sig, "r", driver="family", memb_size=10*1024**3) as f:
        sig_n_trk = f["TauJets/nTracksTotal"][...]
        sig_n_chr_trk = f["TauJets/nTracks"][...]
        sig_n_cls = f["TauJets/nClustersTotal"][...]
        sig_pt = f["TauJets/pt"][...]


    with h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as f:
        bkg_n_trk = f["TauJets/nTracksTotal"][...]
        bkg_n_chr_trk = f["TauJets/nTracks"][...]
        bkg_n_cls = f["TauJets/nClustersTotal"][...]
        bkg_pt = f["TauJets/pt"][...]

    if args.mode1p:
        # Selection
        sig_sel = (sig_n_chr_trk == 1)
        bkg_sel = (bkg_n_chr_trk == 1)

        # pt-reweight
        sw, bw = pt_reweight(sig_pt[sig_sel], bkg_pt[bkg_sel])

        # nTracks Plot 1P
        nbins = 50
        bins = np.arange(0, nbins + 4, 2) - 1

        fig, ax = overlay(sig_n_trk[sig_sel], sw,
                          bkg_n_trk[bkg_sel], bw,
                          bins=bins)
        ax.set_xlim(0, nbins)
        ax.set_xlabel("Number of tracks", ha="right", x=1.0)
        ax.set_ylabel("Normalised number of events", ha="right", y=1.0)

        fig.savefig("ntrk_1p.pdf")

        # nCluster Plot 1P
        nbins = 30
        bins = np.arange(0, nbins + 2, 1) - 0.5

        fig, ax = overlay(sig_n_cls[sig_sel], sw,
                          bkg_n_cls[bkg_sel], bw,
                          bins=bins)
        ax.set_xlim(0, nbins)
        ax.set_xlabel("Number of clusters", ha="right", x=1.0)
        ax.set_ylabel("Normalised number of events", ha="right", y=1.0)

        fig.savefig("ncls_1p.pdf")
    elif args.mode3p:
        # Selection
        sig_sel = (sig_n_chr_trk == 3)
        bkg_sel = (bkg_n_chr_trk == 3)

        # pt-reweight
        sw, bw = pt_reweight(sig_pt[sig_sel], bkg_pt[bkg_sel])

        # nTracks Plot 3P
        nbins = 50
        bins = np.arange(0, nbins + 4, 2) - 1

        fig, ax = overlay(sig_n_trk[sig_sel], sw,
                          bkg_n_trk[bkg_sel], bw,
                          bins=bins)
        ax.set_xlim(0, nbins)
        ax.set_xlabel("Number of tracks", ha="right", x=1.0)
        ax.set_ylabel("Normalised number of events", ha="right", y=1.0)

        fig.savefig("ntrk_3p.pdf")

        # nCluster Plot 3P
        nbins = 30
        bins = np.arange(0, nbins + 2, 1) - 0.5

        fig, ax = overlay(sig_n_cls[sig_sel], sw,
                          bkg_n_cls[bkg_sel], bw,
                          bins=bins)
        ax.set_xlim(0, nbins)
        ax.set_xlabel("Number of clusters", ha="right", x=1.0)
        ax.set_ylabel("Normalised number of events", ha="right", y=1.0)

        fig.savefig("ncls_3p.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode1p", action="store_true")
    group.add_argument("--mode3p", action="store_true")

    args = parser.parse_args()
    main(args)
