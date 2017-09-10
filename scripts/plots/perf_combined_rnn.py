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

prefix_1p = "/lustre/user/cdeutsch/rnn_tauid/combined/1p/track_cluster_mlp/test"
sig_deco_1p = os.path.join(prefix_1p, "sig_pred.h5")
bkg_deco_1p = os.path.join(prefix_1p, "bkg_pred.h5")

prefix_3p = "/lustre/user/cdeutsch/rnn_tauid/combined/3p/track_cluster_mlp/test"
sig_deco_3p = os.path.join(prefix_3p, "sig_pred.h5")
bkg_deco_3p = os.path.join(prefix_3p, "bkg_pred.h5")


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
                         branches=[branchname, "weight"])
    bkg_bdt = root2array(fbkg, treename="CollectionTree",
                         branches=[branchname, "weight"])

    y = np.concatenate([sig_bdt[branchname], bkg_bdt[branchname]])
    y_true = np.concatenate([np.ones(len(sig_bdt)), np.zeros(len(bkg_bdt))])
    w = np.concatenate([sig_bdt["weight"], bkg_bdt["weight"]])

    return y, y_true, w


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

    with h5py.File(fbkg, "r", driver="family", memb_size=10*1024**3) as f:
        # Sanity check
        assert len(f["TauJets/pt"]) == lbkg
        pt_bkg = f["TauJets/pt"][bkg_idx:]

    w_sig, w_bkg = pt_reweight(pt_sig, pt_bkg)

    y = np.concatenate([y_sig, y_bkg])
    y_true = np.concatenate([np.ones_like(y_sig), np.zeros_like(y_bkg)])
    w = np.concatenate([w_sig, w_bkg])

    return y, y_true, w

# BDT
y_bdt_1p, y_true_bdt_1p, w_bdt_1p = load_bdt(fsig_bdt_1p, fbkg_bdt_1p,
                                             branchname_1p)
y_bdt_3p, y_true_bdt_3p, w_bdt_3p = load_bdt(fsig_bdt_3p, fbkg_bdt_3p,
                                             branchname_3p)

# Track-RNN
y_1p, y_true_1p, w_1p = load_rnn(fsig_1p, fbkg_1p, sig_deco_1p, bkg_deco_1p)
y_3p, y_true_3p, w_3p = load_rnn(fsig_3p, fbkg_3p, sig_deco_3p, bkg_deco_3p)

# Calculate ROC
eff_1p, rej_1p = roc(y_true_1p, y_1p, sample_weight=w_1p)
eff_bdt_1p, rej_bdt_1p = roc(y_true_bdt_1p, y_bdt_1p, sample_weight=w_bdt_1p)
interpol_1p = interp1d(eff_bdt_1p, rej_bdt_1p)

eff_3p, rej_3p = roc(y_true_3p, y_3p, sample_weight=w_3p)
eff_bdt_3p, rej_bdt_3p = roc(y_true_bdt_3p, y_bdt_3p, sample_weight=w_bdt_3p)
interpol_3p = interp1d(eff_bdt_3p, rej_bdt_3p)


# Plot ROC
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(1, 1e4)

ax.plot(eff_bdt_1p, rej_bdt_1p, c="C0", label="BDT 1P")
ax.plot(eff_bdt_3p, rej_bdt_3p, c="C3",label="BDT 3P")
ax.plot(eff_1p, rej_1p, c="C1",label="Track-RNN 1P")
ax.plot(eff_3p, rej_3p, c="C2",label="Track-RNN 3P")

ax.set_xlabel("Signal efficiency", ha="right", x=1)
ax.set_ylabel("Rejection", ha="right", y=1)
ax.legend()

fig.savefig("roc.pdf")


# Bootstrap ratios
n_bootstrap = 10
x = np.linspace(0.03, 1.0, 100)

# Bootstrap 1-prongs
ratio_1p = []
for i in tqdm(range(n_bootstrap)):
    idx = np.random.randint(len(y_1p), size=len(y_1p))
    eff, rej = roc(y_true_1p[idx], y_1p[idx], sample_weight=w_1p[idx])
    interpol = interp1d(eff, rej, copy=False)

    ratio_1p.append(interpol(x) / interpol_1p(x))

# Bootstrap 3-prongs
ratio_3p = []
for i in tqdm(range(n_bootstrap)):
    idx = np.random.randint(len(y_3p), size=len(y_3p))
    eff, rej = roc(y_true_3p[idx], y_3p[idx], sample_weight=w_3p[idx])
    interpol = interp1d(eff, rej, copy=False)

    ratio_3p.append(interpol(x) / interpol_3p(x))

ratio_1p = np.array(ratio_1p)
ratio_3p = np.array(ratio_3p)

mean_1p = np.mean(ratio_1p, axis=0)
mean_3p = np.mean(ratio_3p, axis=0)
std_1p = np.std(ratio_1p, axis=0)
std_3p = np.std(ratio_3p, axis=0)

# Plot ratios
fig = plt.figure()
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)

ax0 = plt.subplot(gs[0])
ax0.tick_params(labelbottom="off")
ax0.set_ylabel("Rejection ratio", ha="right", y=1)
ax0.set_xlim(0, 1)
ax0.set_ylim(1.0, 2.5)

ax0.text(0.94, 0.86, "1-prong", ha="right", va="top", fontsize=8,
         transform=ax0.transAxes)

ax0.plot(x, mean_1p, c="r")
ax0.fill_between(x, mean_1p - std_1p, mean_1p + std_1p, facecolor="r", alpha=0.4)

ax1 = plt.subplot(gs[1])
ax1.set_xlabel("Signal efficiency", ha="right", x=1)
#ax1.set_ylabel("Ratio", ha="right", y=1)
ax1.set_xlim(0, 1)
ax1.set_ylim(0.9, 2.0)
#ax1.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
ax1.text(0.94, 0.86, "3-prong", ha="right", va="top", fontsize=8,
         transform=ax1.transAxes)

ax1.plot(x, mean_3p, c="r")
ax1.fill_between(x, mean_3p - std_3p, mean_3p + std_3p, facecolor="r", alpha=0.4)

fig.savefig("ratios.pdf")
