import argparse

import numpy as np
from root_numpy import root2array, list_branches
import h5py
from tqdm import tqdm

from rnn_tauid.common import cuts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile",
                        help="Output file")
    parser.add_argument("selection",
                        choices=["truth1p", "1p", "truth3p", "3p"],
                        help="Selection to apply to the taus")
    parser.add_argument("infiles", nargs="+",
                        help="Input root files with flattened containers")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Padding for unavailable tracks
    default_value = 0

    # Maximum number of tracks
    n_tracks = 20
    treename = "CollectionTree"
    prefix = "TauTracks"
    h5group = "track"
    # Names of derived fields
    add_fields = ["TauTracks.charge"]

    # Get branches
    branches = list_branches(args.infiles[0], treename="CollectionTree")
    branches = [branch for branch in branches if branch.startswith(prefix)]

    # root2array kwargs
    opt_rnp = {
        "selection": cuts.sel_dict[args.selection],
        "treename": treename,
        "branches": [(branch, default_value, n_tracks) for branch in branches]
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
        "fletcher32": True
    }

    # dtype for single track
    dt = np.dtype([(var, np.float32) for var in branches + add_fields])

    # Maximum buffer size in bytes
    buffer_bytes = 512 * 1024**2
    # Chunked read
    chunksize = buffer_bytes // (dt.itemsize * n_tracks)
    chunks = [(i, min(i + chunksize, len_total))
              for i in range(0, len_total, chunksize)]

    buffer = np.empty((chunksize, n_tracks), dtype=dt)

    with h5py.File(args.outfile, "a") as outf:
        ds = outf.create_dataset(h5group, (len_pass, n_tracks),
                                 dtype=dt, **opt_h5)

        write_pos = 0
        for start, stop in tqdm(chunks):
            chunk = root2array(args.infiles, start=start, stop=stop, **opt_rnp)
            chunk_len = len(chunk)

            # Move chunk into buffer
            for var in branches:
                buffer[:chunk_len][var] = chunk[var]

            # Implementation of derived fields
            buffer[:chunk_len]["TauTracks.charge"] = \
                np.sign(chunk["TauTracks.qOverP"])

            ds[write_pos:write_pos+chunk_len] = buffer[:chunk_len]
            write_pos += chunk_len

    # Sanity check
    assert write_pos == len_pass
