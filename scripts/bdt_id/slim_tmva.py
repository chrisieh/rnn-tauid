import argparse
import os
from glob import glob

import h5py


def main(args):
    from root_numpy import root2array

    infiles = glob(args.pattern)
    nfiles = len(infiles)

    for i, fin in enumerate(infiles):
        print("Processing {}/{}: ".format(i + 1, nfiles) + fin)

        # Outfile path
        head, tail = os.path.split(fin)
        fout = os.path.join(head, args.fout)

        # Check if already exists
        if os.path.exists(fout):
            print("Outfile {} already exists - skipping".format(fout))
            continue

        # Do the slimming
        branches = ["classID", "weight", "classifier"]
        data = {}

        for tree in ["TrainTree", "TestTree"]:
            treename = "dataset/" + tree
            data[tree] = root2array(fin, treename=treename, branches=branches)

        with h5py.File(fout, "w") as f:
            for tree in ["TrainTree", "TestTree"]:
                f.create_dataset(tree, data=data[tree], compression="gzip",
                                 compression_opts=9, shuffle=True,
                                 fletcher32=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--fout", default="TMVA_slimmed.h5")

    args = parser.parse_args()
    main(args)
