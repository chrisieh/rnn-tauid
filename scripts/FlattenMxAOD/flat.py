import argparse
import glob
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outfile")
    parser.add_argument("--truth", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pattern = os.path.join(args.indir, "*.root")
    infiles = sorted(glob.glob(pattern))
    outfile = args.outfile

    import ROOT
    from ROOT import TChain, gROOT

    gROOT.ProcessLine(".L AnalysisSelector.C+")

    ch = TChain("CollectionTree")
    for f in infiles:
        ch.Add(f)

    selector = ROOT.AnalysisSelector(outfile, args.truth)
    ch.Process(selector)
