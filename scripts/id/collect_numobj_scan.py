import argparse
from glob import glob
import re

import pandas as pd


def main(args):
    files = glob(args.pattern)

    p = re.compile(r".*_([0-9]*)/.*")
    res = {"num": [], "val_loss_mean": [], "val_loss_std": []}

    for filename in files:
        print("Processing {}".format(filename))

        matches = p.findall(filename)
        assert len(matches) == 1
        num = int(matches[0])

        print("Matched {} number of input objects".format(num))

        log = pd.read_csv(filename).iloc[-10:]
        mean = log.val_loss.mean()
        std = log.val_loss.std()
        print("mean: {}".format(mean))
        print("std: {}".format(std))
        print("")

        res["num"].append(num)
        res["val_loss_mean"].append(mean)
        res["val_loss_std"].append(std)

    df = pd.DataFrame(res)
    df = df.sort_values("num")
    df.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("-o", dest="outfile", default="results.csv")

    args = parser.parse_args()
    main(args)
