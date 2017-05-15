import argparse

import numpy as np
import h5py

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup()

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width
from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


def main(args):
    # Load NN
    with h5py.File(args.sig, "r", driver="family", memb_size=10*1024**3) as f:
        sig_len = len(f["TauJets/pt"])
        sig_idx = int(0.5 * sig_len)
        
        sig_pt = f["TauJets/pt"][sig_idx:]
        sig_mu = f["TauJets/mu"][sig_idx:]

    with h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as f:
        bkg_len = len(f["TauJets/pt"])
        bkg_idx = int(0.5 * bkg_len)

        bkg_pt = f["TauJets/pt"][bkg_idx:]
        bkg_mu = f["TauJets/pt"][bkg_idx:]

    with h5py.File(args.sig_deco, "r") as f:
        sig_y = f["score"][sig_idx:]

    with h5py.File(args.bkg_deco, "r") as f:
        bkg_y = f["score"][bkg_idx:]

    # Load R21-TAUID
    sig_ntuple_fname = "/lustre/user/cdeutsch/Data/R21-Training/sig1P_test_deco.h5"
    bkg_ntuple_fname = "/lustre/user/cdeutsch/Data/R21-Training/bkg1P_test_deco.h5"

    with h5py.File(sig_ntuple_fname, "r") as s, h5py.File(bkg_ntuple_fname, "r") as b:
        r21 = {
            "y_score": np.concatenate([
                s["CollectionTree"]["vars2016_pt_gamma_1p_isofix"],
                b["CollectionTree"]["vars2016_pt_gamma_1p_isofix"]
            ]),
            "y_true": np.concatenate([
                np.ones(len(s["CollectionTree"])),
                np.zeros(len(b["CollectionTree"]))
            ]),
            "weight": np.concatenate([
                s["CollectionTree"]["weight"],
                b["CollectionTree"]["weight"]
            ]),
            "pt": np.concatenate([
                s["CollectionTree"]["TauJets.pt"],
                b["CollectionTree"]["TauJets.pt"]
            ]),
            "mu": np.concatenate([
                s["CollectionTree"]["TauJets.mu"],
                b["CollectionTree"]["TauJets.mu"]
            ]),
            "nVtxPU": np.concatenate([
                s["CollectionTree"]["TauJets.nVtxPU"],
                b["CollectionTree"]["TauJets.nVtxPU"]
            ])
        }

        r21["is_sig"] = r21["y_true"] == 1

    r21_flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
    r21_passes_thr = r21_flat.fit(r21["pt"][r21["is_sig"]],
                                  r21["mu"][r21["is_sig"]],
                                  r21["y_score"][r21["is_sig"]])

    assert np.isclose(np.count_nonzero(r21_passes_thr) / float(len(r21_passes_thr)),
                      args.eff, atol=0, rtol=1e-2)

    flat = Flattener(binnings.pt_flat, binnings.mu_flat, args.eff)
    passes_thr = flat.fit(sig_pt, sig_mu, sig_y)
    
    assert np.isclose(np.count_nonzero(passes_thr) / float(len(passes_thr)),
                      args.eff, atol=0, rtol=1e-2)

    r21_bkg_pass_thr = r21_flat.passes_thr(r21["pt"][~r21["is_sig"]],
                                           r21["mu"][~r21["is_sig"]],
                                           r21["y_score"][~r21["is_sig"]])
    bkg_pass_thr = flat.passes_thr(bkg_pt, bkg_mu, bkg_y)

    bins = 10 ** np.linspace(np.log10(20000), np.log10(200000), 9)
    bin_midpoint = bin_center(bins)
    bin_half_width = bin_width(bins) / 2.0

    # Background efficiency
    r21_bkg_eff = binned_efficiency(r21["pt"][~r21["is_sig"]], r21_bkg_pass_thr, bins=bins)
    bkg_eff = binned_efficiency(bkg_pt, bkg_pass_thr, bins=bins)

    # Background rejection
    r21_bkg_rej = 1.0 / r21_bkg_eff.mean
    d_r21_bkg_rej = r21_bkg_eff.std / r21_bkg_eff.mean ** 2
    
    bkg_rej = 1.0 / bkg_eff.mean
    d_bkg_rej = bkg_eff.std / bkg_eff.mean ** 2

    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(bin_midpoint / 1000.0, bkg_rej,
                xerr=bin_half_width / 1000.0, yerr=d_bkg_rej,
                fmt="o", label="Combined NN")
    ax.errorbar(bin_midpoint / 1000.0, r21_bkg_rej,
                xerr=bin_half_width / 1000.0, yerr=d_r21_bkg_rej,
                fmt="o", label="R21 Tau-ID")
    ax.set_xlim(20, 200)
    #ax.set_ylim(40, 200)
    ax.set_xlabel("pt / GeV", ha="right", x=1.0)
    ax.set_ylabel("Inverse background efficiency", ha="right", y=1.0)
    ax.legend()

    fig.savefig("rej.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")
    parser.add_argument("sig_deco")
    parser.add_argument("bkg_deco")

    parser.add_argument("--eff", type=float, default=0.6)

    args = parser.parse_args()
    main(args)
