import argparse

def main(args):
    import ROOT

    nevents = 0
    for infile in args.infiles:
        f = ROOT.TFile(infile, "READ")
        cutflow = f.Get("CutFlow")
        nevents += cutflow.GetBinContent(1)
        f.Close()

    print("Total events: {}".format(nevents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs="+")
    args = parser.parse_args()
    main(args)
