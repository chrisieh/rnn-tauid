import argparse

import numpy as np
import pandas as pd
import h5py

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width


def main(args):
    with h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as f:
        pt = f["TauJets/pt"][...]
        eta = f["TauJets/eta"][...]
        nTracks = f["TauJets/nTracks"][...]

    sel = (pt > 20000) & ((np.abs(eta) < 1.37) | (np.abs(eta) > 1.52)) \
          & (np.abs(eta) < 2.5)

    pt = pt[sel] / 1000.0
    nTracks = nTracks[sel]

    oneprong = (nTracks == 1)
    threeprong = (nTracks == 3)

    bins = np.linspace(20, 400, 25)

    # if args.mode1p:
    #     pass_trk = oneprong
    # elif args.mode3p:
    #     pass_trk = threeprong
    # else:
    #     # Shouldn't occur
    #     pass_trk = None

    bkg_eff = binned_efficiency(pt, oneprong, bins=bins)
    bkg_eff_2 = binned_efficiency(pt, threeprong, bins=bins)

    rej = 1.0 / bkg_eff.mean
    d_rej = bkg_eff.std / bkg_eff.mean ** 2
    rej_2 = 1.0 / bkg_eff_2.mean
    d_rej_2 = bkg_eff_2.std / bkg_eff_2.mean ** 2

    bin_midpoint = bin_center(bins)
    bin_half_width = bin_width(bins) / 2.0

    fig, ax = plt.subplots()
    ax.errorbar(bin_midpoint, rej, xerr=bin_half_width, yerr=d_rej,
                fmt="o", c="r", label="1-track")
    ax.errorbar(bin_midpoint, rej_2, xerr=bin_half_width, yerr=d_rej_2,
                fmt="o", c="b", label="3-track")
    ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)
    ax.set_ylabel("Rejection", ha="right", y=1.0)

    ax.set_xticks([20, 100, 200, 300, 400])

    ylim = ax.get_ylim()
    ax.set_ylim(0, 65)

    ax.legend()

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bkg")
    parser.add_argument("-o", dest="outfile", required=True)

    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--mode1p", action="store_true")
    # group.add_argument("--mode3p", action="store_true")

    args = parser.parse_args()
    main(args)
