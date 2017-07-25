import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.32, aspect_ratio=0.7, pad_left=0.31, pad_bottom=0.12)
mpl.rcParams["xtick.major.size"] = 4
mpl.rcParams["xtick.minor.size"] = 2
mpl.rcParams["ytick.major.size"] = 4
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
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
        "ptIntermediateAxis": r"$p_\mathrm{T}^\mathrm{clamp}$",
        "trFlightPathSig": r"$S_\mathrm{T}^\mathrm{flight}$",
        "massTrkSys": r"$m_\mathrm{track}$",
        "dRmax": r"$\Delta R_\mathrm{max}$"
    }

    df = pd.read_csv(args.infile)

    df = df.sort_values("ratio", ascending=False)
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.axvline(0, linestyle="dotted", c="#999999", linewidth=0.6, zorder=-1)
    ax.errorbar(100 - 100 * df.ratio, df.index.values, xerr=100 * df.dratio,
                fmt="o", color="b")
    ax.set_xlim(-5, 75)
    ax.set_ylim(df.index.min() - 0.75, df.index.max() + 0.75)
    ax.set_yticks(df.index.values)
    ax.set_yticklabels([var2tex[v] for v in df["var"]])
    ax.set_xlabel("Rejection loss / %", ha="right", x=1)
    ax.tick_params(axis="y", which="minor", left="off", right="off")

    # Set text labels
    for idx in df.index:
        entry = df.iloc[idx]

        x = 100 - 100 * entry.ratio
        if x >= 37.5:
            x -= 5
            ha = "right"
        else:
            x += 5
            ha = "left"

        # Round error to one decimal place (in percent)
        err = np.ceil(10 * 100 * entry.dratio) / 10.0
        s = r"$({:.1f} \pm {:.1f})\,\%$".format(100 - 100 * entry.ratio, err)
        ax.text(x, idx - 0.06, s, ha=ha, va="center", fontsize=7)

    fig.savefig("importance.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")

    args = parser.parse_args()
    main(args)
