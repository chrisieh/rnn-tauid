import argparse

import numpy as np
import h5py
from tqdm import tqdm

from rnn_tauid.common import cuts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile",
                        help="Output file")
    parser.add_argument("selection",
                        choices=["truth1p", "1p",
                                 "truth3p", "3p",
                                 "truthXp", "Xp"],
                        help="Selection to apply to the taus")
    parser.add_argument("infiles", nargs="+",
                        help="Input root files with flattened containers")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Import here to avoid root taking over the command line
    from root_numpy import root2array, list_branches

    # Padding for unavailable tracks
    default_value = 0

    # Maximum number of tracks
    n_clusters = 10
    treename = "CollectionTree"
    prefix = "TauClusters"
    h5group = "cluster"

    # Get branches
    branches = list_branches(args.infiles[0], treename="CollectionTree")
    branches = [branch for branch in branches if branch.startswith(prefix)]

    # root2array kwargs
    opt_rnp = {
        "selection": cuts.sel_dict[args.selection],
        "treename": treename,
        "branches": [(branch, default_value, n_clusters)
                     for branch in branches]
    }

    # Total length and length after selection for chunking and preallocation
    len_total = len(root2array(args.infiles, treename=treename,
                               branches=(branches[0], default_value, 1)))
    len_pass = len(root2array(args.infiles, treename=treename,
                              selection=opt_rnp["selection"],
                              branches=(branches[0], default_value, 1)))

    # h5py.Dataset kwargs
    opt_h5 = {
        "compression": "gzip",
        "compression_opts": 9,
        "shuffle": True,
        "fletcher32": True
    }

    # dtype for single cluster
    dt = np.dtype([(var, np.float32) for var in branches])

    # Maximum buffer size in bytes
    buffer_bytes = 512 * 1024**2
    # Chunked read
    chunksize = buffer_bytes // (dt.itemsize * n_clusters)
    chunks = [(i, min(i + chunksize, len_total))
              for i in range(0, len_total, chunksize)]

    buffer = np.empty((chunksize, n_clusters), dtype=dt)

    with h5py.File(args.outfile, "a", driver="family", memb_size=10*1024**3) as outf:
        ds = outf.create_dataset(h5group, (len_pass, n_clusters),
                                 dtype=dt, **opt_h5)

        write_pos = 0
        for start, stop in tqdm(chunks):
            chunk = root2array(args.infiles, start=start, stop=stop, **opt_rnp)
            chunk_len = len(chunk)

            # Move chunk into buffer
            for var in branches:
                buffer[:chunk_len][var] = chunk[var]

            # Set clusters with all zeros to nan
            view = buffer[:chunk_len].view(np.float32).reshape(
                buffer[:chunk_len].shape + (-1,))
            all_zero = np.all(view == 0, axis=2, keepdims=True)
            mask = np.broadcast_to(all_zero, view.shape)
            view[mask] = np.nan

            ds[write_pos:write_pos+chunk_len] = buffer[:chunk_len]
            write_pos += chunk_len

    # Sanity check
    assert write_pos == len_pass
