import argparse

import numpy as np
import h5py

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)
import matplotlib.pyplot as plt

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width


def main(args):
    with h5py.File(args.data, "r", driver="family", memb_size=10*1024**3) as f:
        pt = f["TauJets/pt"]
        idx = int(0.5 * len(pt))

        n_tracks = f["TauJets/nTracks"][idx:]
        mask = (n_tracks == 1) | (n_tracks == 3)

        pt = pt[idx:][mask]
        truth = f["TauJets/truthDecayMode"][idx:][mask]

        if args.pantau:
            pred = f["TauJets/PanTau_DecayMode"][idx:][mask]

    if not args.pantau:
        with h5py.File(args.deco, "r") as f:
            pred = f["score"][idx:][mask]
            pred = np.argmax(pred, axis=1)

    if args.highpt:
        #pt_bins = np.linspace(20.0, 250.0, 25) #np.logspace(np.log10(100.0), np.log10(1000.0), 21)
        pt_bins = np.linspace(20.0, 300.0, 25)
    else:
        pt_bins = np.linspace(20.0, 100.0, 25)

    pt_bin_center = bin_center(pt_bins)
    pt_bin_halfwidth = 0.5 * bin_width(pt_bins)

    fig, ax = plt.subplots()

    for i, mode in enumerate([r"$h^\pm$", r"$h^\pm \pi^0$",
                              r"$h^\pm \geq 2 \pi^0$", r"$3 h^\pm$",
                              r"$3 h^\pm \geq 1 \pi^0$"]):
        if args.purity:
            is_mode = pred == i
            passes = truth[is_mode] == i
        else:
            is_mode = truth == i
            passes = pred[is_mode] == i

        pt_mode = pt[is_mode] / 1000.0

        eff = binned_efficiency(pt_mode, passes, bins=pt_bins)

        ax.errorbar(pt_bin_center, eff.mean,
                    xerr=pt_bin_halfwidth, yerr=eff.std,
                    fmt="o", label=mode)

    ax.legend(ncol=2, columnspacing=0.1)

    if args.purity:
        ax.set_ylim(0.68, 0.93)
    elif args.highpt:
        ax.set_ylim(0.4, 1.0)
    else:
        lo, hi = ax.get_ylim()
        # ax.set_ylim(lo, 1.0)
        ax.set_ylim(0.4, 1.0)

    if args.purity and args.highpt:
        ax.set_ylim(0.65, 0.93)

    ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)

    if args.purity:
        ax.set_ylabel(r"Purity", ha="right", y=1.0)
    else:
        ax.set_ylabel(r"Efficiency", ha="right", y=1.0)

    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("deco")
    parser.add_argument("-o", dest="out", default="efficiency_profile.pdf")
    parser.add_argument("--pantau", action="store_true")
    parser.add_argument("--purity", action="store_true")
    parser.add_argument("--highpt", action="store_true")

    args = parser.parse_args()
    main(args)
