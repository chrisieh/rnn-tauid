import argparse

import numpy as np
import h5py


def main(args):
    # Binary cross entropy: L = -t log(p) - (1-t) log(1 - p) (p: prediction,
    # t: target)

    with h5py.File(args.sig_deco, "r") as f:
        sig_p = f["score"][...]

    with h5py.File(args.bkg_deco, "r") as f:
        bkg_p = f["score"][...]

    sig_t = np.ones_like(sig_p)
    bkg_t = np.zeros_like(bkg_p)

    p = np.concatenate([sig_p, bkg_p])
    t = np.concatenate([sig_t, bkg_t])

    ce = -t * np.log(p) - (1. - t) * np.log(1. - p)

    print(np.mean(ce))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig_deco")
    parser.add_argument("bkg_deco")

    args = parser.parse_args()
    main(args)
