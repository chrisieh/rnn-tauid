import argparse

import numpy as np
import h5py
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="File to shuffle")
    parser.add_argument("outfile", help="Shuffled output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with h5py.File(args.infile, "r") as infile, \
            h5py.File(args.outfile, "w") as outfile:
        # Length of datasets for compatibility check
        ds_len = None

        # Loop over datasets in root group
        for name, data_in in tqdm(infile.items(), desc="Dataset"):
            if not isinstance(data_in, h5py.Dataset):
                continue

            # Compatibility check
            if ds_len:
                assert(len(data_in) == ds_len)
            ds_len = len(data_in)

            data_out = outfile.create_dataset(
                name, data_in.shape, dtype=data_in.dtype, compression="gzip",
                fletcher32=True)

            if len(data_in.shape) == 1:
                # If array one dimensional shuffle everything at once
                it = [Ellipsis]
            elif len(data_in.shape) == 2:
                # For two dimensions iterate over the last dimension
                i, j = data_in.shape
                it = range(j)
            else:
                raise RuntimeError("Dataset incompatible for shuffling")

            for j in tqdm(it, desc="Column"):
                # Reset random state every time to ensure identical shuffling
                random_state = np.random.RandomState(seed=0)
                col = data_in[:, j]
                random_state.shuffle(col)
                data_out[:, j] = col

    print("Shuffled {} entries".format(ds_len))
