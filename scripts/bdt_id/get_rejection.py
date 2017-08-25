import argparse

import h5py
import numpy as np
from tqdm import tqdm


def main(args):
    with h5py.File(args.infile, "r") as f:
        y_true = f["TestTree"]["classID"]
        y = f["TestTree"]["classifier"]
        w = f["TestTree"]["weight"]

        # Flip y_true labels
        is_sig = y_true == 0
        y_true[is_sig] = 1
        y_true[~is_sig] = 0

    rej = []

    for i in tqdm(range(100)):
        idx = np.random.randint(len(y), size=len(y))

        y_bs = y[idx]
        is_sig_bs = is_sig[idx]
        w_bs = w[idx]

        p = np.percentile(y_bs[is_sig_bs], 100.0 - args.eff)
        bkg_pass = np.sum(w_bs[(~is_sig_bs) & (y_bs > p)])
        bkg_total = np.sum(w_bs[~is_sig_bs])

        rej.append(bkg_total / bkg_pass)

    rej = np.array(rej)
    print("Rej.: {} +- {}".format(rej.mean(), rej.std()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("--eff", type=float, default=60.0)

    args = parser.parse_args()
    main(args)
