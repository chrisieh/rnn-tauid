import argparse

import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()

from rnn_tauid.evaluation.misc import bin_center, bin_width


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
    from collections import namedtuple
    from root_numpy import root2array
    import os

    Var = namedtuple("Var", ["bins", "func", "label"])

    variables_1p = {
        "centFrac": Var(
            bins=np.linspace(0.0, 1.0, 23), func=None,
            label=r"$f_\mathrm{cent}$"
        ),
        "etOverPtLeadTrk": Var(
            bins=np.linspace(0.0, 20.0, 23), func=None,
            label=r"$f^{-1}_\mathrm{leadtrack}$"
        ),
        "innerTrkAvgDist": Var(
            bins=np.linspace(0.0, 0.4, 23), func=None,
            label=r"$R^{0.2}_\mathrm{track}$"
        ),
        "absipSigLeadTrk": Var(
            bins=np.linspace(0.0, 15.0, 23), func=None,
            label=r"$\left| S_\mathrm{leadtrack} \right|$"
        ),
        "SumPtTrkFrac": Var(
            bins=np.linspace(0.0, 1.0, 23), func=None,
            label=r"$f^\mathrm{track}_\mathrm{iso}$"
        ),
        "ChPiEMEOverCaloEME": Var(
            bins=np.linspace(-2.0, 2.0, 23), func=None,
            label=r"$f^\mathrm{track-HAD}_\mathrm{EM}$"
        ),
        "EMPOverTrkSysP": Var(
            bins=np.linspace(0.0, 30.0, 23), func=None,
            label=r"$f^\mathrm{EM}_\mathrm{track}$"
        ),
        "ptRatioEflowApprox": Var(
            bins=np.linspace(0.0, 2.3, 23), func=None,
            label=r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$"
        ),
        "mEflowApprox": Var(
            bins=np.linspace(0.0, 10.0, 23),
            func=lambda x: x / 1000.0,
            label=r"$m_\mathrm{EM+track}$ / GeV"
        ),
        "massTrkSys": Var(
            bins=np.linspace(0.0, 25.0, 23),
            func=lambda x: x / 1000.0,
            label=r"$m_\mathrm{track}$ / GeV"
        )
    }

    variables_3p = {
        "centFrac": Var(
            bins=np.linspace(0.0, 1.0, 23), func=None,
            label=r"$f_\mathrm{cent}$"
        ),
        "etOverPtLeadTrk": Var(
            bins=np.linspace(0.0, 12.0, 23), func=None,
            label=r"$f^{-1}_\mathrm{leadtrack}$"
        ),
        "innerTrkAvgDist": Var(
            bins=np.linspace(0.0, 0.4, 23), func=None,
            label=r"$R^{0.2}_\mathrm{track}$"
        ),
        "SumPtTrkFrac": Var(
            bins=np.linspace(0.0, 1.0, 23), func=None,
            label=r"$f^\mathrm{track}_\mathrm{iso}$"
        ),
        "ChPiEMEOverCaloEME": Var(
            bins=np.linspace(-1.5, 2.5, 23), func=None,
            label=r"$f^\mathrm{track-HAD}_\mathrm{EM}$"
        ),
        "EMPOverTrkSysP": Var(
            bins=np.linspace(0.0, 8.0, 23), func=None,
            label=r"$f^\mathrm{EM}_\mathrm{track}$"
        ),
        "ptRatioEflowApprox": Var(
            bins=np.linspace(0.0, 2.5, 23), func=None,
            label=r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$"
        ),
        "mEflowApprox": Var(
            bins=np.linspace(0.0, 10.0, 23),
            func=lambda x: x / 1000.0,
            label=r"$m_\mathrm{EM+track}$ / GeV"
        ),
        "dRmax": Var(
            bins=np.linspace(0.0, 0.4, 23), func=None,
            label=r"$\Delta R_\mathrm{max}$"
        ),
        "trFlightPathSig": Var(
            bins=np.linspace(0.0, 20.0, 23), func=None,
            label=r"$S^\mathrm{flight}_\mathrm{T}$"
        ),
        "massTrkSys": Var(
            bins=np.linspace(0.0, 15.0, 23),
            func=lambda x: x / 1000.0,
            label=r"$m_\mathrm{track}$ / GeV"
        )
    }

    variables_1p_transf = {
        "centFrac": Var(
            bins=np.linspace(0.0, 1.0, 23),
            func=lambda x: np.minimum(x, 1),
            label=r"$f_\mathrm{cent}$"
        ),
        "etOverPtLeadTrk": Var(
            bins=np.linspace(-1.5, 5, 23),
            func=lambda x: np.log(np.maximum(0.1, x)),
            label=r"$f^{-1}_\mathrm{leadtrack}$"
        ),
        "innerTrkAvgDist": Var(
            bins=np.linspace(0.0, 0.4, 23),
            func=None,
            label=r"$R^{0.2}_\mathrm{track}$"
        ),
        "absipSigLeadTrk": Var(
            bins=np.linspace(0.0, 15.0, 23),
            func=lambda x: np.minimum(x, 30),
            label=r"$\left| S_\mathrm{leadtrack} \right|$"
        ),
        "SumPtTrkFrac": Var(
            bins=np.linspace(0.0, 1.0, 23),
            func=None,
            label=r"$f^\mathrm{track}_\mathrm{iso}$"
        ),
        "ChPiEMEOverCaloEME": Var(
            bins=np.linspace(-3.0, 3.0, 23),
            func=lambda x: np.maximum(-4, np.minimum(x, 5)),
            label=r"$f^\mathrm{track-HAD}_\mathrm{EM}$"
        ),
        "EMPOverTrkSysP": Var(
            bins=np.linspace(-2, 2.5, 23),
            func=lambda x: np.log10(np.maximum(1e-2, x)),
            label=r"$f^\mathrm{EM}_\mathrm{track}$"
        ),
        "ptRatioEflowApprox": Var(
            bins=np.linspace(0.0, 2.3, 23),
            func=lambda x: np.minimum(x, 4),
            label=r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$"
        ),
        "mEflowApprox": Var(
            bins=np.linspace(np.log10(140), 5, 23),
            func=lambda x: np.log10(np.maximum(140, x)),
            label=r"$m_\mathrm{EM+track}$ / MeV"
        ),
        "massTrkSys": Var(
            bins=np.linspace(np.log10(140), 5, 23),
            func=lambda x: np.log10(np.maximum(140, x)),
            label=r"$m_\mathrm{track}$ / MeV"
        )
    }

    variables_3p_transf = {
        "centFrac": Var(
            bins=np.linspace(0.0, 1.0, 23),
            func=lambda x: np.minimum(x, 30),
            label=r"$f_\mathrm{cent}$"
        ),
        "etOverPtLeadTrk": Var(
            bins=np.linspace(-1.5, 5, 23),
            func=lambda x: np.log(np.maximum(0.1, x)),
            label=r"$f^{-1}_\mathrm{leadtrack}$"
        ),
        "innerTrkAvgDist": Var(
            bins=np.linspace(0.0, 0.4, 23),
            func=None,
            label=r"$R^{0.2}_\mathrm{track}$"
        ),
        "SumPtTrkFrac": Var(
            bins=np.linspace(0.0, 1.0, 23),
            func=None,
            label=r"$f^\mathrm{track}_\mathrm{iso}$"
        ),
        "ChPiEMEOverCaloEME": Var(
            bins=np.linspace(-3.0, 3.0, 23),
            func=lambda x: np.maximum(-4, np.minimum(x, 5)),
            label=r"$f^\mathrm{track-HAD}_\mathrm{EM}$"
        ),
        "EMPOverTrkSysP": Var(
            bins=np.linspace(-2, 2.5, 23),
            func=lambda x: np.log10(np.maximum(1e-2, x)),
            label=r"$f^\mathrm{EM}_\mathrm{track}$"
        ),
        "ptRatioEflowApprox": Var(
            bins=np.linspace(0.0, 2.3, 23),
            func=lambda x: np.minimum(x, 4),
            label=r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$"
        ),
        "mEflowApprox": Var(
            bins=np.linspace(np.log10(140), 5, 23),
            func=lambda x: np.log10(np.maximum(140, x)),
            label=r"$m_\mathrm{EM+track}$ / MeV"
        ),
        "dRmax": Var(
            bins=np.linspace(0.0, 0.4, 23),
            func=None,
            label=r"$\Delta R_\mathrm{max}$"
        ),
        "trFlightPathSig": Var(
            bins=np.linspace(0.0, 20.0, 23),
            func=lambda x: np.log10(np.maximum(0.01, x)),
            label=r"$S^\mathrm{flight}_\mathrm{T}$"
        ),
        "massTrkSys": Var(
            bins=np.linspace(np.log10(140), 5, 23),
            func=lambda x: np.log10(np.maximum(140, x)),
            label=r"$m_\mathrm{track}$ / MeV"
        )
    }

    if args.oneprong:
        if args.transform:
            variables = variables_1p_transf
        else:
            variables = variables_1p
    elif args.threeprong:
        if args.transform:
            variables = variables_3p_transf
        else:
            variables = variables_3p

    # Read weight
    rnp_opt = dict(
        treename="CollectionTree"
    )
    sw = root2array(args.sig, branches="weight", **rnp_opt)
    bw = root2array(args.bkg, branches="weight", **rnp_opt)
    for var in tqdm(variables):
        # Read variable
        s = root2array(args.sig, branches="TauJets." + var, **rnp_opt)
        b = root2array(args.bkg, branches="TauJets." + var, **rnp_opt)

        # Apply transformation
        if variables[var].func:
            s = variables[var].func(s)
            b = variables[var].func(b)

        # Plot
        fig, ax = overlay(s, sw, b, bw, bins=variables[var].bins)

        if variables[var].label:
            ax.set_xlabel(variables[var].label, ha="right", x=1)
        else:
            ax.set_xlabel(var, ha="right", x=1)

        ax.set_ylabel("Tau candidates", ha="right", y=1)

        fig.savefig(os.path.join(args.outdir, var + ".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("--outdir", dest="outdir", default="")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--1p", dest="oneprong", action="store_true")
    group.add_argument("--3p", dest="threeprong", action="store_true")

    parser.add_argument("--transform", action="store_true")

    args = parser.parse_args()
    main(args)
