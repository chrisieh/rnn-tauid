import argparse

import numpy as np
import matplotlib as mpl
mpl.use("PDF")
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(args):
    epoch, _, train_loss, _, val_loss = np.loadtxt(
        "log.csv", delimiter=",", skiprows=1, unpack=True)
    epoch = epoch.astype(np.int32)

    fig, ax = plt.subplots()
    ax.plot(epoch + 1, train_loss, label="Training loss")
    ax.plot(epoch + 1, val_loss, label="Validation loss")
    ax.set_xlim(0, np.max(epoch + 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()

    fig.savefig(args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("out")

    args = parser.parse_args()
    main(args)
