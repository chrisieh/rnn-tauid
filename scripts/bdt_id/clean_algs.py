import argparse
import os
from glob import glob


def main(args):
    algs = glob(args.pattern)

    for alg in algs:
        tmva = os.path.join(alg, "aux", "TMVA.root")
        slimmed = os.path.join(alg, "aux", "TMVA_slimmed.h5")

        # Skip if no TMVA.root or no TMVA_slimmed.h5 exists
        if not os.path.exists(tmva) or not os.path.exists(slimmed):
            continue

        # Else remove TMVA.root since slimmed version exists
        print("Removing: {}".format(tmva))
        os.remove(tmva)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")

    args = parser.parse_args()
    main(args)
