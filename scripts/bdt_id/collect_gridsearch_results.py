import argparse
import os
import json
from glob import glob

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp


def main(args):
    algs = glob(args.pattern)
    n_algs = len(algs)

    data = {"name": [], "train_time": [], "MaxDepth": [], "MinNodeSize": [],
            "NTrees": [], "Shrinkage": [], "AdaBoostBeta": []}

    metrics = {"roc_auc": [], "roc_auc_train": [], "ks_pval_sig": [],
               "ks_pval_bkg": []}

    # Efficiencies to evaluate
    efficiencies = [5 * i for i in range(9, 20)]
    for eff in efficiencies:
        metrics["eff" + str(eff)] = []
        metrics["eff" + str(eff) + "_train"] = []

    for i, alg in enumerate(algs):
        print("Processing {}/{}: ".format(i + 1, n_algs) + alg)

        config_json = os.path.join(alg, "config.json")
        assert os.path.exists(config_json)

        with open(config_json, "r") as f:
            config = json.load(f)

        # Only process trained algs
        if "trained" not in config["info"] or not config["info"]["trained"]:
            print("Alg is not trained - skipping")
            continue

        # Ensure that TMVA_slimmed.h5 exists
        tmva_slimmed = os.path.join(alg, "aux", "TMVA_slimmed.h5")
        if not os.path.exists(tmva_slimmed):
            print("TMVA_slimmed.h5 does not exist - skipping")
            continue

        # Read from config json
        data["name"].append(config["name"])
        data["train_time"].append(config["info"]["train_time"])

        opts = config["config"]["algopts"]
        for algopt in ["MaxDepth", "MinNodeSize", "NTrees", "Shrinkage",
                       "AdaBoostBeta"]:
            if algopt in opts:
                data[algopt].append(opts[algopt])
            else:
                data[algopt].append(None)

        # Read data from TMVA_slimmed.h5
        with h5py.File(tmva_slimmed, "r") as f:
            y_true = f["TestTree"]["classID"]
            y = f["TestTree"]["classifier"]
            w = f["TestTree"]["weight"]

            y_true_train = f["TrainTree"]["classID"]
            y_train = f["TrainTree"]["classifier"]
            w_train = f["TrainTree"]["weight"]

        # Flip y_true labels
        is_sig = y_true == 0
        y_true[is_sig] = 1
        y_true[~is_sig] = 0

        is_sig_train = y_true_train == 0
        y_true_train[is_sig_train] = 1
        y_true_train[~is_sig_train] = 0

        # ROC-AUC testing & training
        roc_auc = roc_auc_score(y_true, y, sample_weight=w)
        roc_auc_train = roc_auc_score(y_true_train, y_train,
                                      sample_weight=w_train)
        metrics["roc_auc"].append(roc_auc)
        metrics["roc_auc_train"].append(roc_auc_train)

        # ROC testing
        fpr, tpr, thr = roc_curve(y_true, y, sample_weight=w)
        fpr_nonzero = fpr != 0
        roc_eff = tpr[fpr_nonzero]
        roc_rej = 1.0 / fpr[fpr_nonzero]
        roc = interp1d(roc_eff, roc_rej, copy=False, assume_sorted=True)

        # ROC training
        fpr_train, tpr_train, thr_train = roc_curve(y_true_train, y_train,
                                                    sample_weight=w_train)
        fpr_train_nonzero = fpr_train != 0
        roc_eff_train = tpr_train[fpr_train_nonzero]
        roc_rej_train = 1.0 / fpr_train[fpr_train_nonzero]
        roc_train = interp1d(roc_eff_train, roc_rej_train, copy=False,
                             assume_sorted=True)

        for eff in efficiencies:
            try:
                metrics["eff" + str(eff)].append(roc(eff / 100.0))
            except ValueError:
                print("Error interpolating roc at {} eff. - test".format(eff))
                print("Min. eff-test: {}".format(roc_eff.min()))
                metrics["eff" + str(eff)].append(None)

            try:
                metrics["eff" + str(eff) + "_train"].append(roc_train(eff / 100.0))
            except ValueError:
                print("Error interpolating roc at {} eff. - train".format(eff))
                print("Min. eff-train: {}".format(roc_eff_train.min()))
                metrics["eff" + str(eff) + "_train"].append(None)

        # KS-Test
        ks_sig = ks_2samp(y[y_true == 1], y_train[y_true_train == 1])
        ks_bkg = ks_2samp(y[y_true == 0], y_train[y_true_train == 0])

        metrics["ks_pval_sig"].append(ks_sig.pvalue)
        metrics["ks_pval_bkg"].append(ks_bkg.pvalue)

    config_df = pd.DataFrame(data)
    metrics_df = pd.DataFrame(metrics)

    df = pd.concat([config_df, metrics_df], axis=1)
    df.to_csv(args.fout, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--fout", default="results_gridscan.csv")

    args = parser.parse_args()
    main(args)
