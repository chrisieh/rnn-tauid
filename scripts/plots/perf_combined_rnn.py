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
mpl_setup(scale=0.48)

from rnn_tauid.common.preprocessing import pt_reweight
from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width
from rnn_tauid.evaluation.flattener import Flattener
import rnn_tauid.common.binnings as binnings


def roc(*args, **kwargs):
    fpr, tpr, thr = roc_curve(*args, **kwargs)
    nonzero = fpr != 0
    eff = tpr[nonzero]
    rej = 1.0 / fpr[nonzero]

    return eff, rej

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

# Calculate ROC
eff_trk_cls_mlp_1p, rej_trk_cls_mlp_1p = roc(y_true_1p, y_trk_cls_mlp_1p,
                                             sample_weight=w_1p)
eff_trk_mlp_1p, rej_trk_mlp_1p = roc(y_true_1p, y_trk_mlp_1p,
                                     sample_weight=w_1p)

eff_trk_cls_mlp_3p, rej_trk_cls_mlp_3p = roc(y_true_3p, y_trk_cls_mlp_3p,
                                             sample_weight=w_3p)
eff_trk_mlp_3p, rej_trk_mlp_3p = roc(y_true_3p, y_trk_mlp_3p,
                                     sample_weight=w_3p)


# For BDT
eff_bdt_1p, rej_bdt_1p = roc(y_true_bdt_1p, y_bdt_1p, sample_weight=w_bdt_1p)
interpol_1p = interp1d(eff_bdt_1p, rej_bdt_1p)
eff_bdt_3p, rej_bdt_3p = roc(y_true_bdt_3p, y_bdt_3p, sample_weight=w_bdt_3p)
interpol_3p = interp1d(eff_bdt_3p, rej_bdt_3p)

# Plot ROC
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(1, 1e4)

ax.plot(eff_bdt_1p, rej_bdt_1p, c="C0", label="BDT 1P")
ax.plot(eff_bdt_3p, rej_bdt_3p, c="C3", label="BDT 3P")
ax.plot(eff_trk_cls_mlp_1p, rej_trk_cls_mlp_1p, c="C1", label="RNN 1P (Track + Cluster + MLP)")
ax.plot(eff_trk_cls_mlp_3p, rej_trk_cls_mlp_3p, c="C2", label="RNN 3P (Track + Cluster + MLP)")
#ax.plot(eff_trk_mlp_1p, rej_trk_mlp_1p, c="C4", label="Trk+MLP 1P")
#ax.plot(eff_trk_mlp_3p, rej_trk_mlp_3p, c="C5", label="Trk+MLP 3P")

ax.set_xlabel("Signal efficiency", ha="right", x=1)
ax.set_ylabel("Rejection", ha="right", y=1)
ax.legend()

fig.savefig("roc.pdf")


# Bootstrap ratios
n_bootstrap = 100
x = np.linspace(0.02, 1.0, 100)

# Bootstrap 1-prongs
ratio_trk_cls_mlp_1p = []
ratio_trk_mlp_1p = []
for i in tqdm(range(n_bootstrap)):
    idx = np.random.randint(len(w_1p), size=len(w_1p))

    y_true = y_true_1p[idx]
    w = w_1p[idx]
    y_trk_cls_mlp = y_trk_cls_mlp_1p[idx]
    y_trk_mlp = y_trk_mlp_1p[idx]

    # Trk+Cls+MLP
    eff, rej = roc(y_true, y_trk_cls_mlp, sample_weight=w)
    interpol = interp1d(eff, rej, copy=False)
    ratio_trk_cls_mlp_1p.append(interpol(x) / interpol_1p(x))

    # Trk+MLP
    eff, rej = roc(y_true, y_trk_mlp, sample_weight=w)
    interpol = interp1d(eff, rej, copy=False)
    ratio_trk_mlp_1p.append(interpol(x) / interpol_1p(x))


# Bootstrap 3-prongs
ratio_trk_cls_mlp_3p = []
ratio_trk_mlp_3p = []
for i in tqdm(range(n_bootstrap)):
    idx = np.random.randint(len(w_3p), size=len(w_3p))

    y_true = y_true_3p[idx]
    w = w_3p[idx]
    y_trk_cls_mlp = y_trk_cls_mlp_3p[idx]
    y_trk_mlp = y_trk_mlp_3p[idx]

    # Trk+Cls+MLP
    eff, rej = roc(y_true, y_trk_cls_mlp, sample_weight=w)
    interpol = interp1d(eff, rej, copy=False)
    ratio_trk_cls_mlp_3p.append(interpol(x) / interpol_3p(x))

    # Trk+MLP
    eff, rej = roc(y_true, y_trk_mlp, sample_weight=w)
    interpol = interp1d(eff, rej, copy=False)
    ratio_trk_mlp_3p.append(interpol(x) / interpol_3p(x))


ratio_trk_cls_mlp_1p = np.array(ratio_trk_cls_mlp_1p)
ratio_trk_mlp_1p = np.array(ratio_trk_mlp_1p)
ratio_trk_cls_mlp_3p = np.array(ratio_trk_cls_mlp_3p)
ratio_trk_mlp_3p = np.array(ratio_trk_mlp_3p)

mean_trk_cls_mlp_1p = np.mean(ratio_trk_cls_mlp_1p, axis=0)
mean_trk_mlp_1p = np.mean(ratio_trk_mlp_1p, axis=0)
mean_trk_cls_mlp_3p = np.mean(ratio_trk_cls_mlp_3p, axis=0)
mean_trk_mlp_3p = np.mean(ratio_trk_mlp_3p, axis=0)

std_trk_cls_mlp_1p = np.std(ratio_trk_cls_mlp_1p, axis=0)
std_trk_mlp_1p = np.std(ratio_trk_mlp_1p, axis=0)
std_trk_cls_mlp_3p = np.std(ratio_trk_cls_mlp_3p, axis=0)
std_trk_mlp_3p = np.std(ratio_trk_mlp_3p, axis=0)


# Plot ratios
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection ratio", ha="right", y=1)
ax0.set_xlim(0, 1)
ax0.set_ylim(0.9, 2.6)
ax0.set_yticks([1.0, 1.5, 2.0, 2.5])

ax0.text(0.94, 0.88, "1-prong", ha="right", va="top", fontsize=8,
         transform=ax0.transAxes)

ax0.plot(x, mean_trk_cls_mlp_1p, c="r", label="RNN (Track + Cluster + MLP)")
ax0.fill_between(x, mean_trk_cls_mlp_1p - std_trk_cls_mlp_1p, mean_trk_cls_mlp_1p + std_trk_cls_mlp_1p, facecolor="r", alpha=0.4)
ax0.plot(x, mean_trk_mlp_1p, c="b", label="RNN (Track + MLP)")
ax0.fill_between(x, mean_trk_mlp_1p - std_trk_mlp_1p, mean_trk_mlp_1p + std_trk_mlp_1p, facecolor="b", alpha=0.4)
ax0.legend(loc="lower left")



ax1 = plt.subplot(gs[1])
ax1.set_xlabel("Signal efficiency", ha="right", x=1)
#ax1.set_ylabel("Ratio", ha="right", y=1)
ax1.set_xlim(0, 1)
ax1.set_ylim(0.9, 2.1)
ax1.set_yticks([1.0, 1.5, 2.0])
ax1.text(0.94, 0.88, "3-prong", ha="right", va="top", fontsize=8,
         transform=ax1.transAxes)

ax1.plot(x, mean_trk_cls_mlp_3p, c="r")
ax1.fill_between(x, mean_trk_cls_mlp_3p - std_trk_cls_mlp_3p, mean_trk_cls_mlp_3p + std_trk_cls_mlp_3p, facecolor="r", alpha=0.4)
ax1.plot(x, mean_trk_mlp_3p, c="b")
ax1.fill_between(x, mean_trk_mlp_3p - std_trk_mlp_3p, mean_trk_mlp_3p + std_trk_mlp_3p, facecolor="b", alpha=0.4)


fig.savefig("ratios.pdf")


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

# RNN 1-prong
is_sig_1p = y_true_1p == 1

flat_1p = Flattener(binnings.pt_flat, binnings.mu_flat, 0.6)
sig_pass_1p = flat_1p.fit(pt_1p[is_sig_1p], mu_1p[is_sig_1p],
                          y_trk_cls_mlp_1p[is_sig_1p])

assert np.isclose(np.count_nonzero(sig_pass_1p) / float(len(sig_pass_1p)),
                  0.6, atol=0, rtol=1e-2)

bkg_pass_1p = flat_1p.passes_thr(pt_1p[~is_sig_1p], mu_1p[~is_sig_1p],
                                 y_trk_cls_mlp_1p[~is_sig_1p])

bkg_eff_1p = binned_efficiency(pt_1p[~is_sig_1p], bkg_pass_1p,bins=bins)
bkg_rej_1p = 1.0 / bkg_eff_1p.mean
d_bkg_rej_1p = bkg_eff_1p.std / bkg_eff_1p.mean ** 2


ratio_1p = bkg_rej_1p / bkg_rej_bdt_1p
d_ratio_1p = d_bkg_rej_1p / bkg_rej_bdt_1p


# Plotting
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])

ax0.set_xlim(20, pt_max)
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection", ha="right", y=1.0)
ax0.set_ylim(50, 300)

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_bdt_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_bdt_1p,
             fmt="o", c="k")

ax0.errorbar(bin_midpoint / 1000.0, bkg_rej_1p,
             xerr=bin_half_width / 1000.0, yerr=d_bkg_rej_1p,
             fmt="o", c="r")


ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel("Ratio")
ax1.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
ax1.set_ylim(1.5, 3.0)

ax1.errorbar(bin_midpoint / 1000.0, ratio_1p,
             xerr=bin_half_width / 1000.0, yerr=d_ratio_1p,
             fmt="o", c="r")

fig.savefig("rnn_1p.pdf")
