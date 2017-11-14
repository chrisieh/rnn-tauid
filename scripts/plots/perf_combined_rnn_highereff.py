import os

import numpy as np
import h5py
from tqdm import tqdm
from root_numpy import root2array
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18, pad_bottom=0.12, aspect_ratio=1.0)

from rnn_tauid.common.preprocessing import pt_reweight
from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width
from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


# RNN-Stuff
fsig_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig1P_v08_%d.h5"
fbkg_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg1P_v08_%d.h5"
fsig_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig3P_v08_%d.h5"
fbkg_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg3P_v08_%d.h5"

# Track + Cluster + MLP
prefix_1p = "/lustre/user/cdeutsch/rnn_tauid/combined/1p/track_cluster_mlp"
sig_deco_trk_cls_mlp_1p = os.path.join(prefix_1p, "sig_pred.h5")
bkg_deco_trk_cls_mlp_1p = os.path.join(prefix_1p, "bkg_pred.h5")

prefix_3p = "/lustre/user/cdeutsch/rnn_tauid/combined/3p/track_cluster_mlp"
sig_deco_trk_cls_mlp_3p = os.path.join(prefix_3p, "sig_pred.h5")
bkg_deco_trk_cls_mlp_3p = os.path.join(prefix_3p, "bkg_pred.h5")

# Track + MLP
prefix_1p = "/lustre/user/cdeutsch/rnn_tauid/combined/1p/track_mlp"
sig_deco_trk_mlp_1p = os.path.join(prefix_1p, "sig_pred.h5")
bkg_deco_trk_mlp_1p = os.path.join(prefix_1p, "bkg_pred.h5")

prefix_3p = "/lustre/user/cdeutsch/rnn_tauid/combined/3p/track_mlp"
sig_deco_trk_mlp_3p = os.path.join(prefix_3p, "sig_pred.h5")
bkg_deco_trk_mlp_3p = os.path.join(prefix_3p, "bkg_pred.h5")

# BDT comparison
prefix_bdt_1p = "/lustre/user/cdeutsch/bdt_tauid/variable_importance/1p/iter_1/decorated_ntuples"
fsig_bdt_1p = os.path.join(prefix_bdt_1p, "sig1P_test_ref.root")
fbkg_bdt_1p = os.path.join(prefix_bdt_1p, "bkg1P_weight_test_ref.root")
branchname_1p = "ChPiEMEOverCaloEME"

prefix_bdt_3p = "/lustre/user/cdeutsch/bdt_tauid/variable_importance/3p/iter_2/decorated_ntuples"
fsig_bdt_3p = os.path.join(prefix_bdt_3p, "sig3P_test_ref.root")
fbkg_bdt_3p = os.path.join(prefix_bdt_3p, "bkg3P_weight_test_ref.root")
branchname_3p = "ChPiEMEOverCaloEME"


def load_bdt(fsig, fbkg, branchname):
    sig_bdt = root2array(fsig, treename="CollectionTree",
                         branches=[branchname, "TauJets.pt", "TauJets.mu", "weight"])
    bkg_bdt = root2array(fbkg, treename="CollectionTree",
                         branches=[branchname, "TauJets.pt", "TauJets.mu", "weight"])

    y = np.concatenate([sig_bdt[branchname], bkg_bdt[branchname]])
    y_true = np.concatenate([np.ones(len(sig_bdt)), np.zeros(len(bkg_bdt))])
    w = np.concatenate([sig_bdt["weight"], bkg_bdt["weight"]])
    pt = np.concatenate([sig_bdt["TauJets.pt"], bkg_bdt["TauJets.pt"]])
    mu = np.concatenate([sig_bdt["TauJets.mu"], bkg_bdt["TauJets.mu"]])

    return y, y_true, w, pt, mu


def load_rnn(fsig, fbkg, sig_deco, bkg_deco):
    # Load decoration
    with h5py.File(sig_deco, "r") as f:
        lsig = len(f["score"])
        sig_idx = int(0.5 * lsig)
        y_sig = f["score"][sig_idx:]

    with h5py.File(bkg_deco, "r") as f:
        lbkg = len(f["score"])
        bkg_idx = int(0.5 * lbkg)
        y_bkg = f["score"][bkg_idx:]

    # Load from original sample
    with h5py.File(fsig, "r", driver="family", memb_size=10*1024**3) as f:
        # Sanity check
        assert len(f["TauJets/pt"]) == lsig
        pt_sig = f["TauJets/pt"][sig_idx:]
        mu_sig = f["TauJets/mu"][sig_idx:]


    with h5py.File(fbkg, "r", driver="family", memb_size=10*1024**3) as f:
        # Sanity check
        assert len(f["TauJets/pt"]) == lbkg
        pt_bkg = f["TauJets/pt"][bkg_idx:]
        mu_bkg = f["TauJets/mu"][bkg_idx:]

    w_sig, w_bkg = pt_reweight(pt_sig, pt_bkg)

    y = np.concatenate([y_sig, y_bkg])
    y_true = np.concatenate([np.ones_like(y_sig), np.zeros_like(y_bkg)])
    w = np.concatenate([w_sig, w_bkg])
    pt = np.concatenate([pt_sig, pt_bkg])
    mu = np.concatenate([mu_sig, mu_bkg])

    return y, y_true, w, pt, mu


# BDT
y_bdt_1p, y_true_bdt_1p, w_bdt_1p, pt_bdt_1p, mu_bdt_1p = load_bdt(
    fsig_bdt_1p, fbkg_bdt_1p, branchname_1p)
y_bdt_3p, y_true_bdt_3p, w_bdt_3p, pt_bdt_3p, mu_bdt_3p = load_bdt(
    fsig_bdt_3p, fbkg_bdt_3p, branchname_3p)

# Trk+Cls+MLP
y_trk_cls_mlp_1p, y_true_1p, w_1p, pt_1p, mu_1p = load_rnn(
    fsig_1p, fbkg_1p, sig_deco_trk_cls_mlp_1p, bkg_deco_trk_cls_mlp_1p)
y_trk_cls_mlp_3p, y_true_3p, w_3p, pt_3p, mu_3p = load_rnn(
    fsig_3p, fbkg_3p, sig_deco_trk_cls_mlp_3p, bkg_deco_trk_cls_mlp_3p)

# Trk+MLP
y_trk_mlp_1p, _, _, _, _ = load_rnn(
    fsig_1p, fbkg_1p, sig_deco_trk_mlp_1p, bkg_deco_trk_mlp_1p)
y_trk_mlp_3p, _, _, _, _ = load_rnn(
    fsig_3p, fbkg_3p, sig_deco_trk_mlp_3p, bkg_deco_trk_mlp_3p)



##### TIGHT #####
# Rejection vs pt
bins = 8
pt_max = 200

bins = 10 ** np.linspace(np.log10(20000), np.log10(pt_max * 1000.0), bins + 1)
bin_midpoint = bin_center(bins)
bin_half_width = bin_width(bins) / 2.0

# BDT 1-prong
bdt_1p_is_sig = y_true_bdt_1p == 1

flat_bdt_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.6)
sig_bdt_pass_1p = flat_bdt_1p.fit(pt_bdt_1p[bdt_1p_is_sig], mu_bdt_1p[bdt_1p_is_sig],
                                  y_bdt_1p[bdt_1p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_1p) / float(len(sig_bdt_pass_1p)),
                  0.6, atol=0, rtol=1e-2)

bkg_bdt_pass_1p = flat_bdt_1p.passes_thr(pt_bdt_1p[~bdt_1p_is_sig], mu_bdt_1p[~bdt_1p_is_sig],
                                         y_bdt_1p[~bdt_1p_is_sig])

bkg_eff_bdt_1p = binned_efficiency(pt_bdt_1p[~bdt_1p_is_sig], bkg_bdt_pass_1p, bins=bins)
bkg_rej_bdt_1p = 1.0 / bkg_eff_bdt_1p.mean
d_bkg_rej_bdt_1p = bkg_eff_bdt_1p.std / bkg_eff_bdt_1p.mean ** 2

# BDT 3-prong
bdt_3p_is_sig = y_true_bdt_3p == 1

flat_bdt_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.45)
sig_bdt_pass_3p = flat_bdt_3p.fit(pt_bdt_3p[bdt_3p_is_sig], mu_bdt_3p[bdt_3p_is_sig],
                                  y_bdt_3p[bdt_3p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_3p) / float(len(sig_bdt_pass_3p)),
                  0.45, atol=0, rtol=1e-2)

bkg_bdt_pass_3p = flat_bdt_3p.passes_thr(pt_bdt_3p[~bdt_3p_is_sig], mu_bdt_3p[~bdt_3p_is_sig],
                                         y_bdt_3p[~bdt_3p_is_sig])

bkg_eff_bdt_3p = binned_efficiency(pt_bdt_3p[~bdt_3p_is_sig], bkg_bdt_pass_3p, bins=bins)
bkg_rej_bdt_3p = 1.0 / bkg_eff_bdt_3p.mean
d_bkg_rej_bdt_3p = bkg_eff_bdt_3p.std / bkg_eff_bdt_3p.mean ** 2

# RNN 1-prong
is_sig_1p = y_true_1p == 1

flat_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.7)
sig_pass_1p = flat_1p.fit(pt_1p[is_sig_1p], mu_1p[is_sig_1p],
                          y_trk_cls_mlp_1p[is_sig_1p])

assert np.isclose(np.count_nonzero(sig_pass_1p) / float(len(sig_pass_1p)),
                  0.7, atol=0, rtol=1e-2)

bkg_pass_1p = flat_1p.passes_thr(pt_1p[~is_sig_1p], mu_1p[~is_sig_1p],
                                 y_trk_cls_mlp_1p[~is_sig_1p])

bkg_eff_1p = binned_efficiency(pt_1p[~is_sig_1p], bkg_pass_1p,bins=bins)
bkg_rej_1p = 1.0 / bkg_eff_1p.mean
d_bkg_rej_1p = bkg_eff_1p.std / bkg_eff_1p.mean ** 2


ratio_1p = bkg_rej_1p / bkg_rej_bdt_1p
d_ratio_1p = d_bkg_rej_1p / bkg_rej_bdt_1p

# RNN 3-prong
is_sig_3p = y_true_3p == 1

flat_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.5)
sig_pass_3p = flat_3p.fit(pt_3p[is_sig_3p], mu_3p[is_sig_3p],
                          y_trk_cls_mlp_3p[is_sig_3p])

assert np.isclose(np.count_nonzero(sig_pass_3p) / float(len(sig_pass_3p)),
                  0.5, atol=0, rtol=1e-2)

bkg_pass_3p = flat_3p.passes_thr(pt_3p[~is_sig_3p], mu_3p[~is_sig_3p],
                                 y_trk_cls_mlp_3p[~is_sig_3p])

bkg_eff_3p = binned_efficiency(pt_3p[~is_sig_3p], bkg_pass_3p,bins=bins)
bkg_rej_3p = 1.0 / bkg_eff_3p.mean
d_bkg_rej_3p = bkg_eff_3p.std / bkg_eff_3p.mean ** 2


ratio_3p = bkg_rej_3p / bkg_rej_bdt_3p
d_ratio_3p = d_bkg_rej_3p / bkg_rej_bdt_3p


# Plotting 1-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 170)

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_1p,
             fmt="o", c="k", label="Optimised BDT\n60% signal efficiency WP")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_1p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)\n70% signal efficiency WP")

ax0.legend(loc="lower right", fontsize=7)

# ax0.text(0.06, 0.94, "60% signal efficiency working point",
#          va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(0.9, 1.6)

ax1.errorbar(bin_midpoint / 1000.0, ratio_1p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_1p,
             fmt="o", c="r")

fig.savefig("rnn_tight_1p.pdf")


# Plotting 3-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 10000)
ax0.set_yticks([0, 2000, 4000, 6000, 8000])

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_3p,
             fmt="o", c="k", label="Optimised BDT\n45% signal efficiency WP")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_3p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)\n50% signal efficiency WP")

ax0.legend(loc="upper left", fontsize=7)

# ax0.text(0.06, 0.94, "45% signal efficiency working point",
#          va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(0.9, 1.9)

ax1.errorbar(bin_midpoint / 1000.0, ratio_3p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_3p,
             fmt="o", c="r")

fig.savefig("rnn_tight_3p.pdf")



###### MEDIUM ###########
# Rejection vs pt
bins = 8
pt_max = 200

bins = 10 ** np.linspace(np.log10(20000), np.log10(pt_max * 1000.0), bins + 1)
bin_midpoint = bin_center(bins)
bin_half_width = bin_width(bins) / 2.0

# BDT 1-prong
bdt_1p_is_sig = y_true_bdt_1p == 1

flat_bdt_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.75)
sig_bdt_pass_1p = flat_bdt_1p.fit(pt_bdt_1p[bdt_1p_is_sig], mu_bdt_1p[bdt_1p_is_sig],
                                  y_bdt_1p[bdt_1p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_1p) / float(len(sig_bdt_pass_1p)),
                  0.75, atol=0, rtol=1e-2)

bkg_bdt_pass_1p = flat_bdt_1p.passes_thr(pt_bdt_1p[~bdt_1p_is_sig], mu_bdt_1p[~bdt_1p_is_sig],
                                         y_bdt_1p[~bdt_1p_is_sig])

bkg_eff_bdt_1p = binned_efficiency(pt_bdt_1p[~bdt_1p_is_sig], bkg_bdt_pass_1p, bins=bins)
bkg_rej_bdt_1p = 1.0 / bkg_eff_bdt_1p.mean
d_bkg_rej_bdt_1p = bkg_eff_bdt_1p.std / bkg_eff_bdt_1p.mean ** 2

# BDT 3-prong
bdt_3p_is_sig = y_true_bdt_3p == 1

flat_bdt_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.6)
sig_bdt_pass_3p = flat_bdt_3p.fit(pt_bdt_3p[bdt_3p_is_sig], mu_bdt_3p[bdt_3p_is_sig],
                                  y_bdt_3p[bdt_3p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_3p) / float(len(sig_bdt_pass_3p)),
                  0.6, atol=0, rtol=1e-2)

bkg_bdt_pass_3p = flat_bdt_3p.passes_thr(pt_bdt_3p[~bdt_3p_is_sig], mu_bdt_3p[~bdt_3p_is_sig],
                                         y_bdt_3p[~bdt_3p_is_sig])

bkg_eff_bdt_3p = binned_efficiency(pt_bdt_3p[~bdt_3p_is_sig], bkg_bdt_pass_3p, bins=bins)
bkg_rej_bdt_3p = 1.0 / bkg_eff_bdt_3p.mean
d_bkg_rej_bdt_3p = bkg_eff_bdt_3p.std / bkg_eff_bdt_3p.mean ** 2

# RNN 1-prong
is_sig_1p = y_true_1p == 1

flat_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.75)
sig_pass_1p = flat_1p.fit(pt_1p[is_sig_1p], mu_1p[is_sig_1p],
                          y_trk_cls_mlp_1p[is_sig_1p])

assert np.isclose(np.count_nonzero(sig_pass_1p) / float(len(sig_pass_1p)),
                  0.75, atol=0, rtol=1e-2)

bkg_pass_1p = flat_1p.passes_thr(pt_1p[~is_sig_1p], mu_1p[~is_sig_1p],
                                 y_trk_cls_mlp_1p[~is_sig_1p])

bkg_eff_1p = binned_efficiency(pt_1p[~is_sig_1p], bkg_pass_1p,bins=bins)
bkg_rej_1p = 1.0 / bkg_eff_1p.mean
d_bkg_rej_1p = bkg_eff_1p.std / bkg_eff_1p.mean ** 2


ratio_1p = bkg_rej_1p / bkg_rej_bdt_1p
d_ratio_1p = d_bkg_rej_1p / bkg_rej_bdt_1p

# RNN 3-prong
is_sig_3p = y_true_3p == 1

flat_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.6)
sig_pass_3p = flat_3p.fit(pt_3p[is_sig_3p], mu_3p[is_sig_3p],
                          y_trk_cls_mlp_3p[is_sig_3p])

assert np.isclose(np.count_nonzero(sig_pass_3p) / float(len(sig_pass_3p)),
                  0.6, atol=0, rtol=1e-2)

bkg_pass_3p = flat_3p.passes_thr(pt_3p[~is_sig_3p], mu_3p[~is_sig_3p],
                                 y_trk_cls_mlp_3p[~is_sig_3p])

bkg_eff_3p = binned_efficiency(pt_3p[~is_sig_3p], bkg_pass_3p,bins=bins)
bkg_rej_3p = 1.0 / bkg_eff_3p.mean
d_bkg_rej_3p = bkg_eff_3p.std / bkg_eff_3p.mean ** 2


ratio_3p = bkg_rej_3p / bkg_rej_bdt_3p
d_ratio_3p = d_bkg_rej_3p / bkg_rej_bdt_3p


# Plotting 1-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 130)

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_1p,
             fmt="o", c="k", label="Optimised BDT")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_1p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)")

ax0.legend(loc="lower right", fontsize=7)

ax0.text(0.06, 0.94, "75% signal efficiency working point",
         va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(1.5, 2.7)
ax1.set_yticks([1.5, 2.0, 2.5])

ax1.errorbar(bin_midpoint / 1000.0, ratio_1p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_1p,
             fmt="o", c="r")

fig.savefig("rnn_medium_1p.pdf")


# Plotting 3-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 4000)
ax0.set_yticks([0, 1000, 2000, 3000, 4000])

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_3p,
             fmt="o", c="k", label="Optimised BDT")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_3p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)")

ax0.legend(loc="lower right", fontsize=7)

ax0.text(0.06, 0.94, "60% signal efficiency working point",
         va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(1.2, 2.6)

ax1.errorbar(bin_midpoint / 1000.0, ratio_3p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_3p,
             fmt="o", c="r")

fig.savefig("rnn_medium_3p.pdf")


###### LOOSE ###########
# Rejection vs pt
bins = 8
pt_max = 200

bins = 10 ** np.linspace(np.log10(20000), np.log10(pt_max * 1000.0), bins + 1)
bin_midpoint = bin_center(bins)
bin_half_width = bin_width(bins) / 2.0

# BDT 1-prong
bdt_1p_is_sig = y_true_bdt_1p == 1

flat_bdt_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.85)
sig_bdt_pass_1p = flat_bdt_1p.fit(pt_bdt_1p[bdt_1p_is_sig], mu_bdt_1p[bdt_1p_is_sig],
                                  y_bdt_1p[bdt_1p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_1p) / float(len(sig_bdt_pass_1p)),
                  0.85, atol=0, rtol=1e-2)

bkg_bdt_pass_1p = flat_bdt_1p.passes_thr(pt_bdt_1p[~bdt_1p_is_sig], mu_bdt_1p[~bdt_1p_is_sig],
                                         y_bdt_1p[~bdt_1p_is_sig])

bkg_eff_bdt_1p = binned_efficiency(pt_bdt_1p[~bdt_1p_is_sig], bkg_bdt_pass_1p, bins=bins)
bkg_rej_bdt_1p = 1.0 / bkg_eff_bdt_1p.mean
d_bkg_rej_bdt_1p = bkg_eff_bdt_1p.std / bkg_eff_bdt_1p.mean ** 2

# BDT 3-prong
bdt_3p_is_sig = y_true_bdt_3p == 1

flat_bdt_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.75)
sig_bdt_pass_3p = flat_bdt_3p.fit(pt_bdt_3p[bdt_3p_is_sig], mu_bdt_3p[bdt_3p_is_sig],
                                  y_bdt_3p[bdt_3p_is_sig])

assert np.isclose(np.count_nonzero(sig_bdt_pass_3p) / float(len(sig_bdt_pass_3p)),
                  0.75, atol=0, rtol=1e-2)

bkg_bdt_pass_3p = flat_bdt_3p.passes_thr(pt_bdt_3p[~bdt_3p_is_sig], mu_bdt_3p[~bdt_3p_is_sig],
                                         y_bdt_3p[~bdt_3p_is_sig])

bkg_eff_bdt_3p = binned_efficiency(pt_bdt_3p[~bdt_3p_is_sig], bkg_bdt_pass_3p, bins=bins)
bkg_rej_bdt_3p = 1.0 / bkg_eff_bdt_3p.mean
d_bkg_rej_bdt_3p = bkg_eff_bdt_3p.std / bkg_eff_bdt_3p.mean ** 2

# RNN 1-prong
is_sig_1p = y_true_1p == 1

flat_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.85)
sig_pass_1p = flat_1p.fit(pt_1p[is_sig_1p], mu_1p[is_sig_1p],
                          y_trk_cls_mlp_1p[is_sig_1p])

assert np.isclose(np.count_nonzero(sig_pass_1p) / float(len(sig_pass_1p)),
                  0.85, atol=0, rtol=1e-2)

bkg_pass_1p = flat_1p.passes_thr(pt_1p[~is_sig_1p], mu_1p[~is_sig_1p],
                                 y_trk_cls_mlp_1p[~is_sig_1p])

bkg_eff_1p = binned_efficiency(pt_1p[~is_sig_1p], bkg_pass_1p,bins=bins)
bkg_rej_1p = 1.0 / bkg_eff_1p.mean
d_bkg_rej_1p = bkg_eff_1p.std / bkg_eff_1p.mean ** 2


ratio_1p = bkg_rej_1p / bkg_rej_bdt_1p
d_ratio_1p = d_bkg_rej_1p / bkg_rej_bdt_1p

# RNN 3-prong
is_sig_3p = y_true_3p == 1

flat_3p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.75)
sig_pass_3p = flat_3p.fit(pt_3p[is_sig_3p], mu_3p[is_sig_3p],
                          y_trk_cls_mlp_3p[is_sig_3p])

assert np.isclose(np.count_nonzero(sig_pass_3p) / float(len(sig_pass_3p)),
                  0.75, atol=0, rtol=1e-2)

bkg_pass_3p = flat_3p.passes_thr(pt_3p[~is_sig_3p], mu_3p[~is_sig_3p],
                                 y_trk_cls_mlp_3p[~is_sig_3p])

bkg_eff_3p = binned_efficiency(pt_3p[~is_sig_3p], bkg_pass_3p,bins=bins)
bkg_rej_3p = 1.0 / bkg_eff_3p.mean
d_bkg_rej_3p = bkg_eff_3p.std / bkg_eff_3p.mean ** 2


ratio_3p = bkg_rej_3p / bkg_rej_bdt_3p
d_ratio_3p = d_bkg_rej_3p / bkg_rej_bdt_3p


# Plotting 1-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 70)

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_1p,
             fmt="o", c="k", label="Optimised BDT")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_1p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)")

ax0.legend(loc="lower right", fontsize=7)

ax0.text(0.06, 0.94, "85% signal efficiency working point",
         va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(1.5, 2.8)
ax1.set_yticks([1.5, 2.0, 2.5])

ax1.errorbar(bin_midpoint / 1000.0, ratio_1p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_1p,
             fmt="o", c="r")

fig.savefig("rnn_loose_1p.pdf")


# Plotting 3-prong
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(0, 1500)

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_3p,
             fmt="o", c="k", label="Optimised BDT")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_3p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_3p,
             fmt="o", c="r", label="RNN (Track + Cluster + MLP)")

ax0.legend(loc="lower right", fontsize=7)

ax0.text(0.06, 0.94, "75% signal efficiency working point",
         va="top", fontsize=7, transform=ax0.transAxes)


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(1.2, 2.2)

ax1.errorbar(bin_midpoint / 1000.0, ratio_3p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_3p,
             fmt="o", c="r")

fig.savefig("rnn_loose_3p.pdf")
