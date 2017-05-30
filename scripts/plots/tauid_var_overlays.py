import argparse

import numpy as np
import h5py
from scipy.stats import binned_statistic

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()

from rnn_tauid.evaluation.misc import bin_center, bin_width


def overlay(s, sw, b, bw, bins=None):
    sigh, _, _ = binned_statistic(s, sw, statistic="sum", bins=bins)
    bkgh, _, _ = binned_statistic(b, bw, statistic="sum", bins=bins)

    def poisson(arr):
        return np.sqrt(np.sum(arr ** 2))

    dsigh, _, _ = binned_statistic(s, sw, statistic=poisson, bins=bins)
    dbkgh, _, _ = binned_statistic(b, bw, statistic=poisson, bins=bins)

    x = bin_center(bins)
    dx = 0.5 * bin_width(bins)

    # Normalize
    sig_norm = np.sum(sigh * bin_width(bins))
    bkg_norm = np.sum(bkgh * bin_width(bins))

    fig, ax = plt.subplots()
    ax.errorbar(x, sigh / sig_norm, xerr=dx, yerr=dsigh / sig_norm, c="r",
                fmt="o", label="Signal")
    ax.errorbar(x, bkgh / bkg_norm, xerr=dx, yerr=dbkgh / bkg_norm, c="b",
                fmt="o", label="Background")
    ax.legend()

    return fig, ax


def main(args):
    from collections import namedtuple
    Var = namedtuple("Var", ["bins"])

    var = {
        "centFrac": Var(
            bins=np.linspace(0.0, 1.0, 23)
        ),
        "etOverPtLeadTrk": Var(
            bins=np.linspace(0.0, 20.0, 23)
        ),
        "innerTrkAvgDist": Var(
            bins=np.linspace(0.0, 0.4, 23)
        ),
        "absipSigLeadTrk": Var(
            bins=np.linspace(0.0, 20.0, 23)
        ),
        "SumPtTrkFrac": Var(
            bins=np.linspace(0.0, 1.0, 23)
        ),
#        "ChPiEMEOverCaloEME": Var(
#            bins=np.linspace(-8.0, 8.0, 23)
#        ),
        "EMPOverTrkSysP": Var(
            bins=np.linspace(0.0, 10000.0, 23)
        ),
        "ptRatioEflowApprox": Var(
            bins=np.linspace(0.0, 10.0, 23)
        ),
        "mEflowApprox": Var(
            bins=np.linspace(0.0, 30000.0, 23)
        ),
        "dRmax": Var(
            bins=np.linspace(0.0, 0.4, 23)
        ),
        "trFlightPathSig": Var(
            bins=np.linspace(0.0, 20.0, 23)
        ),
        "massTrkSys": Var(
            bins=np.linspace(0.0, 30000.0, 23)
        )
    }

    with h5py.File(args.sig, "r", driver="family", memb_size=10*1024**3) as sig, \
         h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as bkg:
        # TODO: pt reweight

        for v in var:
            s = sig["TauJets/" + v][...]
            sw = np.ones_like(s)
            b = bkg["TauJets/" + v][...]
            bw = np.ones_like(b)

            fig, ax = overlay(s, sw, b, bw, bins=var[v].bins)
            fig.savefig(v + ".pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")

    args = parser.parse_args()
    main(args)
