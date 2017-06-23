import argparse
import os
from array import array

import numpy as np
import pandas as pd


variables = [
    "TauJets.centFrac", "TauJets.etOverPtLeadTrk",
    "TauJets.innerTrkAvgDist", "TauJets.absipSigLeadTrk",
    "TauJets.SumPtTrkFrac", "TauJets.ChPiEMEOverCaloEME",
    "TauJets.EMPOverTrkSysP", "TauJets.ptRatioEflowApprox",
    "TauJets.mEflowApprox"
]

expressions = [
    "TMath::Min(TauJets.centFrac, 1.0)",
    "TMath::Log10(TMath::Max(0.1, TauJets.etOverPtLeadTrk))",
    "TauJets.innerTrkAvgDist",
    "TMath::Min(TauJets.absipSigLeadTrk, 30)",
    "TauJets.SumPtTrkFrac",
    "TMath::Max(-4, TMath::Min(TauJets.ChPiEMEOverCaloEME, 5))",
    "TMath::Log10(TMath::Max(0.01, TauJets.EMPOverTrkSysP))",
    "TMath::Min(TauJets.ptRatioEflowApprox, 4)",
    "TMath::Log10(TMath::Max(140, TauJets.mEflowApprox))"
]

transformations = {
    "TauJets.centFrac": lambda x: np.minimum(x, 1.0),
    "TauJets.etOverPtLeadTrk": lambda x: np.log10(np.maximum(0.1, x)),
    "TauJets.innerTrkAvgDist": lambda x: x,
    "TauJets.absipSigLeadTrk": lambda x: np.minimum(x, 30),
    "TauJets.SumPtTrkFrac": lambda x: x,
    "TauJets.ChPiEMEOverCaloEME": lambda x: np.maximum(-4, np.minimum(x, 5)),
    "TauJets.EMPOverTrkSysP": lambda x: np.log10(np.maximum(0.01, x)),
    "TauJets.ptRatioEflowApprox": lambda x: np.minimum(x, 4),
    "TauJets.mEflowApprox": lambda x: np.log10(np.maximum(140, x))
}


def partial_dependence_tmva(sig, bkg, model, v1_idx, v2_idx, xy, transf=False):
    from ROOT import TMVA
    from root_numpy.tmva import evaluate_reader

    if transf:
        var = expressions
    else:
        var = variables

    X = np.concatenate([sig, bkg])

    reader = TMVA.Reader()
    for v in var:
        reader.AddVariable(v, array("f", [0.]))
    reader.BookMVA("BDT method", model)

    pd = []
    for x, y in xy:
        X[:, v1_idx] = x
        X[:, v2_idx] = y
        pred = evaluate_reader(reader, "BDT method", X)
        pd.append(np.mean(pred))

    return pd


def main(args):
    v1_idx = variables.index(args.var1)
    v2_idx = variables.index(args.var2)

    v1_sampling = np.linspace(args.x_range[0], args.x_range[1], args.x_bins)
    v2_sampling = np.linspace(args.y_range[0], args.y_range[1], args.y_bins)

    if args.transformed:
        branches = expressions
    else:
        branches = variables

    # Load data
    from root_numpy import root2array
    X_sig = root2array(args.sigf, treename="CollectionTree", branches=branches)
    X_bkg = root2array(args.bkgf, treename="CollectionTree", branches=branches)

    # Cast variables if needed
    dtype = [(fname, np.float32) for fname, ftype in X_sig.dtype.descr]
    X_sig = X_sig.astype(dtype, copy=False)
    X_bkg = X_bkg.astype(dtype, copy=False)

    X_sig = X_sig.view(np.float32).reshape(len(X_sig), -1)
    np.random.shuffle(X_sig)
    X_sig = X_sig[:args.num_events, :].copy()

    X_bkg = X_bkg.view(np.float32).reshape(len(X_bkg), -1)
    np.random.shuffle(X_bkg)
    X_bkg = X_bkg[:args.num_events, :].copy()

    # Grid for partial dependence plot
    xx, yy = np.meshgrid(v1_sampling, v2_sampling)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    if args.pbs_id is not None:
        lo = args.pbs_id * args.x_bins
        hi = lo + args.x_bins

        assert args.pbs_id < args.y_bins

        sl = np.s_[lo:hi, ...]
    else:
        sl = np.s_[...]

    # Apply trafo if needed
    if args.transformed:
        f1 = transformations[args.var1]
        f2 = transformations[args.var2]

        grid_trsf = grid.copy()
        grid_trsf[:, 0] = f1(grid[:, 0])
        grid_trsf[:, 1] = f2(grid[:, 1])

        res = partial_dependence_tmva(X_sig, X_bkg, args.model, v1_idx, v2_idx,
                                      grid_trsf[sl], transf=True)
    else:
        res = partial_dependence_tmva(X_sig, X_bkg, args.model, v1_idx, v2_idx,
                                      grid[sl], transf=False)

    df = pd.DataFrame({"x": grid[sl][:, 0], "y": grid[sl][:, 1], "pd": res})
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigf")
    parser.add_argument("bkgf")
    parser.add_argument("model")
    parser.add_argument("var1")
    parser.add_argument("var2")
    parser.add_argument("--transformed", action="store_true")
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--y-range", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--x-bins", type=int, default=100)
    parser.add_argument("--y-bins", type=int, default=100)
    parser.add_argument("-o", dest="out", default="pd_2d_result.csv")
    parser.add_argument("--pbs-id", type=int, help="Maximum same as number of "
                        "y-bins")

    args = parser.parse_args()
    main(args)
