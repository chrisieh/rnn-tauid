import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, aspect_ratio=1.0, pad_left=0.25)
mpl.rcParams["ytick.major.size"] = 4
import matplotlib.pyplot as plt


def main(args):
    var2tex = {
        "centFrac": r"$f_\mathrm{cent}$",
        "etOverPtLeadTrk": r"$f^{-1}_\mathrm{leadtrack}$",
        "innerTrkAvgDist": r"$R_\mathrm{track}$",
        "absipSigLeadTrk": r"$\left| S_\mathrm{leadtrack} \right|$",
        "SumPtTrkFrac": r"$f^\mathrm{track}_\mathrm{iso}$",
        "ChPiEMEOverCaloEME": r"$f^\mathrm{track-HAD}_\mathrm{EM}$",
        "EMPOverTrkSysP": r"$f^\mathrm{EM}_\mathrm{track}$",
        "ptRatioEflowApprox": r"$p^\mathrm{EM+track}_\mathrm{T} / p_\mathrm{T}$",
        "mEflowApprox": r"$m_\mathrm{EM+track}$",
        "ptIntermediateAxis": r"$p_\mathrm{T}^\mathrm{clamp}$"
    }

    dummy_data = {
        "var": ["centFrac", "etOverPtLeadTrk", "innerTrkAvgDist",
                "absipSigLeadTrk", "SumPtTrkFrac", "ChPiEMEOverCaloEME",
                "EMPOverTrkSysP", "ptRatioEflowApprox", "mEflowApprox",
                "ptIntermediateAxis"],
        "loss": [0.8, 0.05, 0.02, 0.3, 0.34, 0.55, 0.25, 0.45, 0.6, 0.1]
    }

    df = pd.DataFrame(dummy_data)

    #Dummy error
    df["dloss"] = 0.1 * df.loss
    df = df.sort_values("loss")
    df = df.reset_index(drop=True)

    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    ax.errorbar(100 * df.loss, df.index.values, xerr=100 * df.dloss, fmt="o")
    ax.set_xlim(0, 100)
    ax.set_ylim(df.index.min() - 0.5, df.index.max() + 0.5)
    ax.set_yticks(df.index.values)
    ax.set_yticklabels([var2tex[v] for v in df["var"]])
    ax.set_xlabel("Rejection loss", ha="right", x=1)
    ax.tick_params(axis="y", which="minor", left="off", right="off")

    fig.savefig("importance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
