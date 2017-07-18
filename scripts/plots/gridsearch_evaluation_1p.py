import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_right=0.93)


def main(args):
    table = pd.read_csv(args.table)
    print(table.head())

    # Unique parameter settings
    uniq_param = {
        "NTrees": np.sort(table.NTrees.unique()),
        "Shrinkage": np.sort(table.Shrinkage.unique()),
        "MaxDepth": np.sort(table.MaxDepth.unique()),
        "MinNodeSize": np.sort(table.MinNodeSize.unique())
    }

    # Axes label
    label_dict = {
        "NTrees": r"Number of trees $N_\mathrm{trees}$",
        "Shrinkage": r"Shrinkage $\eta$",
        "MaxDepth": r"Maximum tree depth $d_\mathrm{tree}$",
        "MinNodeSize": r"Minimum node size $f_\mathrm{node}^\mathrm{min} \, / \, \%$"
    }

    text_dict = {
        "NTrees": "$ N_\\mathrm{{trees}} = {} $",
        "Shrinkage": "$ \\eta = {} $",
        "MaxDepth": "$ d_\\mathrm{{tree}} = {} $",
        "MinNodeSize": "$ f_\\mathrm{{node}}^\\mathrm{{min}} = {}\\%% $"
    }

    # Working points
    wp = [60, 75, 85, 95]

    for eff in wp:
        print("Best rejection for {}% efficiency point: ".format(eff))

        idx_best = table["eff" + str(eff)].argmax()
        print("Index: {}".format(idx_best))
        print(table.iloc[idx_best][["NTrees", "Shrinkage", "MaxDepth",
                                    "MinNodeSize", "eff{}".format(eff),
                                    "eff{}_train".format(eff), "train_time"]])
        print("\n")

    # Parameter histograms
    hists = [
        ("Shrinkage", "NTrees"),
        ("MaxDepth", "NTrees"),
        ("MinNodeSize", "MaxDepth"),
        ("Shrinkage", "MaxDepth")
    ]
    fixed_params = [
        {"MaxDepth": 8, "MinNodeSize": 0.1},
        {"Shrinkage": 0.1, "MinNodeSize": 0.1},
        {"NTrees": 400, "Shrinkage": 0.1},
        {"NTrees": 400, "MinNodeSize": 0.1}
    ]
    zvar = "eff" + str(args.eff)

    for (xvar, yvar), params in zip(hists, fixed_params):
        # Select fixed parameters
        sel = pd.notnull(table.train_time)
        for k, v in params.items():
            sel = sel & (table[k] == v)

        table_sel = table[sel]

        # Build matrix
        xlen, ylen = len(uniq_param[xvar]), len(uniq_param[yvar])
        mat = np.full((ylen, xlen), fill_value=np.nan, dtype=np.float32)

        for i, ival in enumerate(uniq_param[xvar]):
            for j, jval in enumerate(uniq_param[yvar]):
                entry = table_sel[(table_sel[xvar] == ival)
                                  & (table_sel[yvar] == jval)]

                if len(entry) != 1:
                    print("Zero or more than one entry remaining - skipping!")
                    continue

                mat[j, i] = entry[zvar]

        # Plot it
        fig, ax = plt.subplots()
        im = ax.imshow(mat, interpolation="nearest", aspect="auto", origin="lower")

        ax.set_xlabel(label_dict[xvar], ha="right", x=1.0)
        ax.set_ylabel(label_dict[yvar], ha="right", y=1.0)

        xticks = np.arange(len(uniq_param[xvar]))
        yticks = np.arange(len(uniq_param[yvar]))

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.get_xaxis().set_ticks_position("bottom")
        ax.get_yaxis().set_ticks_position("left")

        ax.set_xticklabels(uniq_param[xvar])
        ax.set_yticklabels(uniq_param[yvar])

        ax.minorticks_off()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] + 1)

        # Text labels
        label = []
        for k, v in params.items():
            label.append(text_dict[k].format(v))
        label = ", ".join(label)
        dx = xlim[1] - xlim[0]
        ax.text(-0.4, ylim[1] + 0.5, label, va="center", fontsize=8)

        ax.text(xlim[1] - 0.1, ylim[1] + 0.5, "{} % sig.\nefficiency".format(args.eff),
                va="center", ha="right", fontsize=8)

        # Colorbar
        cb = fig.colorbar(im)
        cb.ax.minorticks_off()
        cb.ax.tick_params(length=4.0)
        cb.set_label("Rejection", ha="right", y=1.0)

        fig.savefig("scan_{x}_{y}.pdf".format(x=xvar, y=yvar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table")
    parser.add_argument("--eff", type=int, default=60)

    args = parser.parse_args()
    main(args)
