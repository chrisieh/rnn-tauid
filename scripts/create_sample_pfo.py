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
                                 "truthXp", "Xp",
                                 "PFO1P", "PFO3P",
                                 "PFOXP"],
                        help="Selection to apply to the taus")
    parser.add_argument("infiles", nargs="+",
                        help="Input root files with flattened containers")

    return parser.parse_args()


# Global config
treename = "CollectionTree"
default_value = 0
n_charged = 6
n_neutral = 12
n_shots = 6
n_hadronic = 6
n_conversion = 8

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
    charged_pfo_branches = [br for br in branches if br.startswith("TauPFOs.charged")]
    neutral_pfo_branches = [br for br in branches if br.startswith("TauPFOs.neutral")]
    shot_pfo_branches = [br for br in branches if br.startswith("TauPFOs.shot")]
    hadronic_pfo_branches = [br for br in branches if br.startswith("TauPFOs.hadronic")]
    conv_trk_branches = [br for br in branches if br.startswith("TauConv.")]

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

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")), data=data,
                                dtype=np.float32, **h5opt)

        # charged PFOs
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauPFOs.chargedPt", default_value, n_charged),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(charged_pfo_branches, desc="Charged PFOs"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_charged),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)

        # neutral PFOs
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauPFOs.neutralPt", default_value, n_neutral),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(neutral_pfo_branches, desc="Neutral PFOs"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_neutral),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)

        # Shot PFOs
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauPFOs.shotPt", default_value, n_shots),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(shot_pfo_branches, desc="Shot PFOs"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_shots),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)


        # Hadronic PFOs
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauPFOs.hadronicPt", default_value, n_hadronic),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(hadronic_pfo_branches, desc="Hadronic PFOs"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_hadronic),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)


        # Conversion tracks
        mask = root2array(args.infiles, treename=treename,
                          branches=("TauConv.pt", default_value, n_conversion),
                          selection=sel)
        mask = mask <= 0

        for br in tqdm(conv_trk_branches, desc="Conversion tracks"):
            data = root2array(args.infiles, treename=treename,
                              branches=(br, default_value, n_conversion),
                              selection=sel)
            data = data.astype(np.float32)

            # Set nan
            data[mask] = np.nan

            # Check if same number of events and shuffle
            if n_events:
                assert n_events == len(data)
            else:
                n_events = len(data)

            #random_state = np.random.RandomState(seed=seed)
            #random_state.shuffle(data)

            outf.create_dataset("{}/{}".format(*br.split(".")),
                                data=data, dtype=np.float32, **h5opt)
