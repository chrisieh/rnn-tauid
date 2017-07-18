import argparse

import numpy as np

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from matplotlib.colors import Normalize, LogNorm


def log_bins():
    pass


def main(args):
    from root_numpy import root2array

    tree = "CollectionTree"
    weight = "weight"

    data = root2array(args.data, treename=tree,
                      branches=[args.xvar, args.yvar, weight])

    # Scale variables
    if args.x_scale:
        data[args.xvar] /= args.x_scale
    if args.y_scale:
        data[args.yvar] /= args.y_scale

    # Create binning
    if args.x_log:
        x_bins = np.logspace(*np.log10(args.x_range), num=args.num_x_bins)
    else:
        x_bins = np.linspace(*args.x_range, num=args.num_x_bins)

    if args.y_log:
        y_bins = np.logspace(*np.log10(args.y_range), num=args.num_y_bins)
    else:
        y_bins = np.linspace(*args.y_range, num=args.num_y_bins)

    # Log z-scale
    norm = LogNorm() if args.z_log else Normalize()

    fig, ax = plt.subplots()
    _, _, _, im = ax.hist2d(data[args.xvar], data[args.yvar],
                            weights=data[weight], bins=[x_bins, y_bins],
                            normed=True, norm=norm, vmin=0)

    # Axes labels
    xlabel = args.x_label if args.x_label else args.xvar
    ylabel = args.y_label if args.y_label else args.yvar
    ax.set_xlabel(xlabel, ha="right", x=1)
    ax.set_ylabel(ylabel, ha="right", y=1)

    cm = fig.colorbar(im)
    cm.ax.minorticks_off()
    cm.ax.tick_params(length=4.0)

    if args.outfile:
        fig.savefig(args.outfile)
    else:
        fig.savefig("_".join([args.xvar, args.yvar]) + ".pdf")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("xvar")
    parser.add_argument("yvar")

    parser.add_argument("--x-range", nargs=2, type=float, required=True)
    parser.add_argument("--y-range", nargs=2, type=float, required=True)
    parser.add_argument("--num-x-bins", type=int, default=20)
    parser.add_argument("--num-y-bins", type=int, default=20)
    parser.add_argument("--x-log", action="store_true")
    parser.add_argument("--y-log", action="store_true")
    parser.add_argument("--z-log", action="store_true")
    parser.add_argument("--x-scale", type=float, default=None)
    parser.add_argument("--y-scale", type=float, default=None)
    parser.add_argument("--x-label", default=None)
    parser.add_argument("--y-label", default=None)

    parser.add_argument("-o", dest="outfile", default=None)

    args = parser.parse_args()
    main(args)
