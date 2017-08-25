import argparse

from root_numpy import root2array, array2root

import numpy as np
from tqdm import tqdm


def main(args):
    treename = "CollectionTree"
    branchname = "TauJets.mcEventNumber"

    infiles = sorted(args.infiles)

    print("Processing files:")
    for fin in infiles:
        print(fin)

    counter = np.uint64(0)
    next_idx = 0
    for fin in infiles:
        print("Processing: {}".format(fin))
        idx = root2array(fin, treename=treename, branches=branchname)

        # Last entry should have highest index
        counter += idx[-1] + np.uint64(1)

        print("Smallest index: {}, Largest index: {}".format(np.min(idx),
                                                             np.max(idx)))
        new_idx = idx + next_idx
        print("New indices from {} to {}".format(np.min(new_idx),
                                                 np.max(new_idx)))

        next_idx = np.max(new_idx) + np.uint64(1)
        print("Next index is: {}".format(next_idx))

        print("Decorating branch...")
        idx_deco = np.array(new_idx, dtype=[("TauJets.eventIndex", np.uint64)])
        print(repr(idx_deco))
        array2root(idx_deco, fin, treename=treename, mode="update")

    # Sanity check
    assert counter == next_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs="+")

    args = parser.parse_args()
    main(args)
