import argparse
from array import array
import os

import numpy as np
from ROOT import TMVA
from root_numpy import root2array
from root_numpy.tmva import evaluate_reader

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18)
import matplotlib.pyplot as plt


def partial_dependence_tmva(sig, bkg, model, var, var_idx, sampling):
    lsig = len(sig)
    lbkg = len(bkg)

    X = np.concatenate([sig, bkg])

    reader = TMVA.Reader()
    for v in var:
        reader.AddVariable(v, array("f", [0.]))
    reader.BookMVA("BDT method", model)

    pd = []

    for x in sampling:
        X[:, var_idx] = x
        pred = evaluate_reader(reader, "BDT method", X)
        pd.append(np.mean(pred))

    return np.array(pd)


def main(args):
    np.random.seed(42)

    if not args.mode3p:
        if not args.trsf:
            variables = ["TauJets.centFrac", "TauJets.etOverPtLeadTrk",
                         "TauJets.innerTrkAvgDist", "TauJets.absipSigLeadTrk",
                         "TauJets.SumPtTrkFrac", "TauJets.ChPiEMEOverCaloEME",
                         "TauJets.EMPOverTrkSysP", "TauJets.ptRatioEflowApprox",
                         "TauJets.mEflowApprox"]
        else:
            variables = ["TMath::Min(TauJets.centFrac, 1.0)",
                         "TMath::Log10(TMath::Max(0.1, TauJets.etOverPtLeadTrk))",
                         "TauJets.innerTrkAvgDist",
                         "TMath::Min(TauJets.absipSigLeadTrk, 30)",
                         "TauJets.SumPtTrkFrac",
                         "TMath::Max(-4, TMath::Min(TauJets.ChPiEMEOverCaloEME, 5))",
                         "TMath::Log10(TMath::Max(0.01, TauJets.EMPOverTrkSysP))",
                         "TMath::Min(TauJets.ptRatioEflowApprox, 4)",
                         "TMath::Log10(TMath::Max(140, TauJets.mEflowApprox))"]

            var_trsf = {
                "TauJets.centFrac": lambda x: np.minimum(x, 1.0),
                "TauJets.etOverPtLeadTrk": lambda x: np.log10(np.maximum(0.1, x)),
                "TauJets.innerTrkAvgDist": lambda x: x,
                "TauJets.absipSigLeadTrk": lambda x: np.minimum(x, 30),
                "TauJets.SumPtTrkFrac": lambda x: x,
                "TauJets.ChPiEMEOverCaloEME": lambda x: np.maximum(-4, np.minimum(x, 5)),
                "TauJets.EMPOverTrkSysP": lambda x: np.log10(np.maximum(0.01, x)),
                "TauJets.ptRatioEflowApprox": lambda x: np.minimum(x, 4),
                "TauJets.mEflowApprox": lambda x: np.log10(np.maximum(140, x))
            }

        var2tex = {
            "TauJets.centFrac": r"$f_\mathrm{cent}$",
            "TauJets.etOverPtLeadTrk": r"$f^{-1}_\mathrm{leadtrack}$",
            "TauJets.innerTrkAvgDist": r"$R^{0.2}_\mathrm{track}$",
            "TauJets.absipSigLeadTrk": r"$\left| S_\mathrm{leadtrack} \right|$",
            "TauJets.SumPtTrkFrac": r"$f^\mathrm{track}_\mathrm{iso}$",
            "TauJets.ChPiEMEOverCaloEME": r"$f^\mathrm{track-HAD}_\mathrm{EM}$",
            "TauJets.EMPOverTrkSysP": r"$f^\mathrm{EM}_\mathrm{track}$",
            "TauJets.ptRatioEflowApprox": r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$",
            "TauJets.mEflowApprox": r"$m_\mathrm{EM+track}$ / GeV"
        }
    else:
        if not args.trsf:
            variables = ["TauJets.centFrac", "TauJets.etOverPtLeadTrk",
                         "TauJets.innerTrkAvgDist", "TauJets.dRmax",
                         "TauJets.trFlightPathSig", "TauJets.massTrkSys",
                         "TauJets.ChPiEMEOverCaloEME",
                         "TauJets.EMPOverTrkSysP", "TauJets.ptRatioEflowApprox",
                         "TauJets.mEflowApprox"]
        else:
            variables = ["TMath::Min(TauJets.centFrac, 1.0)",
                         "TMath::Log10(TMath::Max(0.1, TauJets.etOverPtLeadTrk))",
                         "TauJets.innerTrkAvgDist",
                         "TauJets.dRmax",
                         "TMath::Log10(TMath::Max(0.01, TauJets.trFlightPathSig))",
                         "TMath::Log10(TMath::Max(140, TauJets.massTrkSys))",
                         "TMath::Max(-4, TMath::Min(TauJets.ChPiEMEOverCaloEME, 5))",
                         "TMath::Log10(TMath::Max(0.01, TauJets.EMPOverTrkSysP))",
                         "TMath::Min(TauJets.ptRatioEflowApprox, 4)",
                         "TMath::Log10(TMath::Max(140, TauJets.mEflowApprox))"]

            var_trsf = {
                "TauJets.centFrac": lambda x: np.minimum(x, 1.0),
                "TauJets.etOverPtLeadTrk": lambda x: np.log10(np.maximum(0.1, x)),
                "TauJets.innerTrkAvgDist": lambda x: x,
                "TauJets.dRmax": lambda x: x,
                "TauJets.trFlightPathSig": lambda x: np.log10(np.maximum(0.01, x)),
                "TauJets.massTrkSys": lambda x: np.log10(np.maximum(140, x)),
                "TauJets.ChPiEMEOverCaloEME": lambda x: np.maximum(-4, np.minimum(x, 5)),
                "TauJets.EMPOverTrkSysP": lambda x: np.log10(np.maximum(0.01, x)),
                "TauJets.ptRatioEflowApprox": lambda x: np.minimum(x, 4),
                "TauJets.mEflowApprox": lambda x: np.log10(np.maximum(140, x))
            }

        var2tex = {
            "TauJets.centFrac": r"$f_\mathrm{cent}$",
            "TauJets.etOverPtLeadTrk": r"$f^{-1}_\mathrm{leadtrack}$",
            "TauJets.innerTrkAvgDist": r"$R^{0.2}_\mathrm{track}$",
            "TauJets.dRmax": r"$\Delta R_\mathrm{max}$",
            "TauJets.trFlightPathSig": r"$S_\mathrm{T}^\mathrm{flight}$",
            "TauJets.massTrkSys": r"$m_\mathrm{track}$ / GeV",
            "TauJets.ChPiEMEOverCaloEME": r"$f^\mathrm{track-HAD}_\mathrm{EM}$",
            "TauJets.EMPOverTrkSysP": r"$f^\mathrm{EM}_\mathrm{track}$",
            "TauJets.ptRatioEflowApprox": r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$",
            "TauJets.mEflowApprox": r"$m_\mathrm{EM+track}$ / GeV"
        }


    def find_var(s):
        for i, var in enumerate(variables):
            if s in var:
                return i

        raise ValueError()

    #var_idx = variables.index(args.variable)
    var_idx = find_var(args.variable)
    plot_range = args.plot_range
    num = args.n

    X_sig = root2array(args.sigf, treename="CollectionTree", branches=variables)
    X_bkg = root2array(args.bkgf, treename="CollectionTree", branches=variables)

    # Cast variables if needed
    dtype = [(fname, np.float32) for fname, ftype in X_sig.dtype.descr]
    X_sig = X_sig.astype(dtype, copy=False)
    X_bkg = X_bkg.astype(dtype, copy=False)

    X_sig = X_sig.view(np.float32).reshape(len(X_sig), -1)
    np.random.shuffle(X_sig)
    X_sig = X_sig[:num, :].copy()

    X_bkg = X_bkg.view(np.float32).reshape(len(X_bkg), -1)
    np.random.shuffle(X_bkg)
    X_bkg = X_bkg[:num, :].copy()


    var_sig = root2array(args.sigf, treename="CollectionTree", branches=args.variable)
    var_bkg = root2array(args.bkgf, treename="CollectionTree", branches=args.variable)

    deciles = np.percentile(
        np.concatenate([var_sig, var_bkg])
        , np.arange(10, 100, 10))

    del var_sig, var_bkg

    sampling = np.linspace(plot_range[0], plot_range[1], 100)

    if args.trsf:
        sampling_trsf = var_trsf[args.variable](sampling)

    if not args.trsf:
        means = partial_dependence_tmva(X_sig, X_bkg, args.modelxml, variables,
                                        var_idx, sampling)
    else:
        means = partial_dependence_tmva(X_sig, X_bkg, args.modelxml, variables,
                                        var_idx, sampling_trsf)

    xscale = 1.0
    if "mEflowApprox" in args.variable or "massTrkSys" in args.variable:
        print("Applying xscale 1000")
        xscale = 1000.0

    fig, ax1 = plt.subplots()
    ax1.plot(sampling/xscale, means, c="r", label="Signal")
    ax1.set_ylabel("Partial dependence", ha="right", y=1.0)
    ax1.set_xlabel(var2tex[args.variable], ha="right", x=1.0)

    lo, hi = ax1.get_ylim()
    r = 0.05 * (hi - lo)
    ax1.set_ylim(lo - r, hi + r)

    ax1.set_xlim(sampling.min() / xscale, sampling.max() / xscale)

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    # Slim deciles
    deltas = []
    for l, h in zip(deciles, deciles[1:]):
        deltas.append(h-l)

    # Draw deciles
    print(deciles)
    ax1.vlines(deciles / xscale, *ax1.get_ylim(), linestyles="dotted", colors="#999999",
               linewidths=0.6)

    # for x, d in zip(list(deciles), range(10, 100, 10)):
    #     ax1.text(x, ylim[1] - 0.05 * dy, "{} %".format(d), color="#999999",
    #              ha="left", va="top", rotation=-90, fontsize=7)

    # Set decile labels

    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigf")
    parser.add_argument("bkgf")
    parser.add_argument("modelxml")
    parser.add_argument("variable")
    parser.add_argument("-n", type=int, default=5000)
    parser.add_argument("-o", dest="out", default="partial_dependence.pdf")
    parser.add_argument("--plot-range", nargs=2, type=float, default=(0.0, 1.0))
    parser.add_argument("--sampling", type=int, default=100)
    parser.add_argument("--trsf", action="store_true")
    parser.add_argument("--mode3p", action="store_true")

    args = parser.parse_args()
    main(args)
