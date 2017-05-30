import argparse
import glob
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outdir")
    parser.add_argument("--truth", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pattern = os.path.join(args.indir, "*.root")
    infiles = sorted(glob.glob(pattern))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    import ROOT
    from ROOT import TChain, gROOT
    gROOT.ProcessLine(".L AnalysisSelector.C+")

    for i, f in enumerate(infiles, 1):
        outfile = os.path.join(args.outdir, os.path.basename(f))
        ch = TChain("CollectionTree")
        ch.Add(f)
        selector = ROOT.AnalysisSelector(outfile, args.truth)
        ch.Process(selector)
        print("{}/{}".format(i, len(infiles)))
