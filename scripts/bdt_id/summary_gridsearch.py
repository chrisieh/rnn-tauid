import argparse

import numpy as np
import pandas as pd


def main(args):
    effstr = "eff{}".format(args.eff)
    effstr_train = "eff{}_train".format(args.eff)

    # Read data
    df = pd.read_csv(args.infile)
    df["subsampling"] = False

    if args.infile_subsampling:
        df2 = pd.read_csv(args.infile_subsampling)
        df2["subsampling"] = True

        # Merge with old one
        df = pd.concat([df, df2], ignore_index=True)

    # Columns to print
    pcols = ["MaxDepth", "MinNodeSize", "NTrees", "Shrinkage", "subsampling",
             "name", "train_time", effstr, effstr_train, "ks_pval_sig",
             "ks_pval_bkg", "ratio"]

    print("Selected performance metric: {}".format(effstr))

    # Sort by best performance
    df["ratio"] = df[effstr] / df[effstr].max()
    df.sort_values("ratio", inplace=True, ascending=False)

    print("")
    print("SUMMARY OF BEST PERFORMERS:")
    print("===========================")
    print(df[pcols].head())

    # KS p-value selection
    ks_thr = 0.05
    sel = (df.ks_pval_sig > ks_thr) & (df.ks_pval_bkg > ks_thr)

    print("")
    print("SUMMARY OF BEST PERFORMERS WITH LARGE KS P-VALUE:")
    print("=================================================")
    print(df[sel][pcols].head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("infile_subsampling", default=None)
    parser.add_argument("--eff", type=int, default=60)

    args = parser.parse_args()
    main(args)
