import argparse

from root_numpy import root2array

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


# Global config
treename = "CollectionTree"
default_value = 0
n_tracks = 20
n_clusters = 15


# h5py dataset kwargs
h5opt = {
    "compression": "gzip",
    "compression_opts": 9,
    "shuffle": True,
    "fletcher32": True
}


if __name__ == "__main__":
    args = get_args()

    # Load here to avoid root taking over the command line
    from root_numpy import root2array, list_branches

    # Branches to load
    branches = list_branches(args.infiles[0], treename=treename)
    jet_branches = [br for br in branches if br.startswith("TauJets")]
    track_branches = [br for br in branches if br.startswith("TauTracks")]
    cluster_branches = [br for br in branches if br.startswith("TauClusters")]

    # Tau selection
    sel = cuts.sel_dict[args.selection]

    with h5py.File(args.outfile, "w", driver="family", memb_size=10*1024**3) as outf:
        # Number of events after selection
        n_events = None
        seed = 1234567890

        # Jet
        for br in tqdm(jet_branches, desc="Jets"):
            data = root2array(args.infiles, treename=treename, branches=br,
                              selection=sel)
            data = data.astype(np.float32)

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            random_state = np.random.RandomState(seed=seed)
            random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")), data=data,
                                dtype=np.float32, **h5opt)

        # Track
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauTracks.pt", default_value, n_tracks),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(track_branches, desc="Tracks"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_tracks),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            random_state = np.random.RandomState(seed=seed)
            random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)

        # Cluster
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauClusters.et", default_value, n_clusters),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(cluster_branches, desc="Clusters"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_clusters),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            random_state = np.random.RandomState(seed=seed)
            random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)
