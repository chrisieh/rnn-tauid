import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()
import matplotlib.pyplot as plt


def main(args):
    df = pd.concat([pd.read_csv(csv) for csv in args.csv], ignore_index=True)

    x_gp = np.sort(df.x.unique())
    y_gp = np.sort(df.y.unique())

    xx, yy = np.meshgrid(x_gp, y_gp)

    dx = x_gp[1] - x_gp[0]
    dy = y_gp[1] - y_gp[0]

    df["x_idx"] = np.round((df.x / dx)).astype(np.int32)
    df["y_idx"] = np.round((df.y / dy)).astype(np.int32)

    grid = np.full((len(y_gp), len(x_gp)), fill_value=np.nan, dtype=np.float32)
    grid[df.y_idx.values, df.x_idx.values] = df.pd.values

    import pdb; pdb.set_trace()

    fig, ax = plt.subplots()
    pcol = ax.pcolormesh(xx, yy, grid)
    pcol.set_edgecolor("face")
    ax.set_xlabel("Var1", ha="right", x=1.0)
    ax.set_ylabel("Var2", ha="right", y=1.0)

    fig.colorbar(pcol)

    fig.savefig("test.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")

    args = parser.parse_args()
    main(args)
