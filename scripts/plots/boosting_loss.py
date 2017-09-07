import argparse

import numpy as np

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)


def main(args):
    margin = np.linspace(-1, 1, 400)

    misclass_loss = np.ones_like(margin)
    misclass_loss[margin > 0] = 0

    exp_loss = np.exp(-margin)

    binom_loss = np.log(1 + np.exp(-2 * margin)) 

    fig, ax = plt.subplots()
    ax.plot(margin, misclass_loss, label="Misclassification loss", c="#999999")
    ax.plot(margin, exp_loss, label="Exponential loss", c="r")
    ax.plot(margin, binom_loss, label="Binomial log-likelihood loss", c="b")

    ax.set_xlabel("Margin $y \, f(x)$", ha="right", x=1.0)
    ax.set_ylabel("Loss", ha="right", y=1.0)

    ax.set_ylim(0.0, 3.0)

    ax.legend()

    fig.savefig(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="outfile", default="boosting_loss.pdf")

    args = parser.parse_args()
    main(args)
