import argparse

import numpy as np
import h5py
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("outfile")
    parser.add_argument("--fraction", default=0.1, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with h5py.File(args.infile, "r") as infile:
        # Search for first dataset in h5-file
        for key in infile:
            ds = None
            if isinstance(infile[key], h5py.Dataset):
                ds = infile[key]
                break

        if not ds:
            raise Exception("No dataset found in hdf5 file.")

        dt = ds.dtype
        offset = np.zeros(1, dtype=dt)
        scale = np.ones(1, dtype=dt)

        dt[""]

        np.nanmean
        np.nanmedian
        np.nanpercentile
        np.nanstd

    with h5py.File(args.outfile, "w") as outfile:
        outfile.create_dataset("offset", data=offset)
        outfile.create_dataset("scale", data=scale)
