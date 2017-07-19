import argparse

import numpy as np
from tqdm import tqdm

from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


def get_eff_rej(sig, bkg, eff_target, score):
    # Do the signal efficiency flattening
    flat = Flattener(binnings.pt_flat, binnings.mu_flat, eff_target)
    sig_pass = flat.fit(sig["TauJets.pt"], sig["TauJets.mu"], sig[score])

    eff = np.count_nonzero(sig_pass) / float(len(sig_pass))
    assert np.isclose(eff, eff_target, rtol=1e-2)

    bkg_pass = flat.passes_thr(bkg["TauJets.pt"], bkg["TauJets.mu"], bkg[score])
    rej = np.sum(bkg["weight"]) / np.sum(bkg["weight"][bkg_pass])

    return eff, rej


def main(args):
    from root_numpy import root2array

    # Load data
    treename = "CollectionTree"
    branches = ["TauJets.pt", "TauJets.mu"] + [args.score, "weight"]
    if args.baseline:
        branches += [args.baseline]

    sig = root2array(args.sig, treename=treename, branches=branches)
    bkg = root2array(args.bkg, treename=treename, branches=branches)

    eff_bs = []
    rej_bs = []
    eff_bs_baseline = []
    rej_bs_baseline = []

    for i in tqdm(range(args.n_bootstrap), desc="bootstrap"):
        sig_idx = np.random.randint(len(sig), size=len(sig))
        bkg_idx = np.random.randint(len(bkg), size=len(bkg))

        sig_bs = sig[sig_idx]
        bkg_bs = bkg[bkg_idx]

        eff, rej = get_eff_rej(sig_bs, bkg_bs, args.eff, args.score)
        eff_bs.append(eff)
        rej_bs.append(rej)

        if args.baseline:
            eff, rej = get_eff_rej(sig_bs, bkg_bs, args.eff, args.baseline)
            eff_bs_baseline.append(eff)
            rej_bs_baseline.append(rej)

    eff_bs = np.array(eff_bs)
    rej_bs = np.array(rej_bs)

    eff, d_eff = eff_bs.mean(), eff_bs.std()
    rej, d_rej = rej_bs.mean(), rej_bs.std()

    print("Eff.: {} +- {}".format(eff, d_eff))
    print("Rej.: {} +- {}".format(rej, d_rej))

    # If we have a baseline calculate ratio
    if args.baseline:
        eff_bs_baseline = np.array(eff_bs_baseline)
        rej_bs_baseline = np.array(rej_bs_baseline)

        ratio_bs = rej_bs / rej_bs_baseline
        ratio, d_ratio = ratio_bs.mean(), ratio_bs.std()

        print("Ratio: {} +- {}".format(ratio, d_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("score")

    parser.add_argument("--baseline", default=None)
    parser.add_argument("--eff", type=float, default=0.6)
    parser.add_argument("--n-bootstrap", type=int, default=10)

    args = parser.parse_args()
    main(args)
