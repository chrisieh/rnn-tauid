import argparse

import numpy as np
import pandas as pd


def main(args):
    df = pd.read_csv(args.log).iloc[-args.n:]
    val_loss_mean = df.val_loss.mean()
    val_loss_std = df.val_loss.std()
    val_acc_mean = df.val_acc.mean()
    val_acc_std = df.val_acc.std()

    print("val. loss.: {:.4} +- {:.2}".format(val_loss_mean, val_loss_std))
    print("val. acc.: {:.4} +- {:.2}".format(val_acc_mean, val_acc_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("-n", type=int, default=10)

    args = parser.parse_args()
    main(args)
