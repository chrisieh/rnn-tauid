import os

import numpy as np
import h5py
from tqdm import tqdm

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_left=0.18)

from rnn_tauid.evaluation.misc import pearsonr
from rnn_tauid.common.preprocessing import pt_reweight

# RNN-Stuff
fsig_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig1P_v08_%d.h5"
fbkg_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg1P_v08_%d.h5"
fsig_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig3P_v08_%d.h5"
fbkg_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg3P_v08_%d.h5"

prefix_1p = "/lustre/user/cdeutsch/rnn_tauid/track_only/1p/ntrack_scan/t_10"
sig_deco_1p = os.path.join(prefix_1p, "sig_pred.h5")
bkg_deco_1p = os.path.join(prefix_1p, "bkg_pred.h5")

prefix_3p = "/lustre/user/cdeutsch/rnn_tauid/track_only/3p/ntrack_scan/t_10"
sig_deco_3p = os.path.join(prefix_3p, "sig_pred.h5")
bkg_deco_3p = os.path.join(prefix_3p, "bkg_pred.h5")


def load_rnn(fsig, fbkg, sig_deco, bkg_deco):
    v = {}
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

        pttrack_sig = f["TauTracks/pt"][sig_idx:, :10]
        d0_sig = f["TauTracks/d0"][sig_idx:, :10]
        dR_sig = f["TauTracks/dRJetSeedAxis"][sig_idx:, :10]
        z0_sig = f["TauTracks/z0sinThetaTJVA"][sig_idx:, :10]
        npix_sig = f["TauTracks/nPixelHits"][sig_idx:, :10]

    with h5py.File(fbkg, "r", driver="family", memb_size=10*1024**3) as f:
        # Sanity check
        assert len(f["TauJets/pt"]) == lbkg
        pt_bkg = f["TauJets/pt"][bkg_idx:]

        pttrack_bkg = f["TauTracks/pt"][bkg_idx:, :10]
        d0_bkg = f["TauTracks/d0"][bkg_idx:, :10]
        dR_bkg = f["TauTracks/dRJetSeedAxis"][bkg_idx:, :10]
        z0_bkg = f["TauTracks/z0sinThetaTJVA"][bkg_idx:, :10]
        npix_bkg = f["TauTracks/nPixelHits"][bkg_idx:, :10]

    w_sig, w_bkg = pt_reweight(pt_sig, pt_bkg)

    y = np.concatenate([y_sig, y_bkg])
    y_true = np.concatenate([np.ones_like(y_sig), np.zeros_like(y_bkg)])
    w = np.concatenate([w_sig, w_bkg])

    v["pttrack"] = np.concatenate([pttrack_sig, pttrack_bkg])
    v["d0"] = np.concatenate([d0_sig, d0_bkg])
    v["dR"] = np.concatenate([dR_sig, dR_bkg])
    v["z0"] = np.concatenate([z0_sig, z0_bkg])
    v["pix"] = np.concatenate([npix_sig, npix_bkg])

    return y, y_true, w, v

# Track-RNN
y_1p, y_true_1p, w_1p, v_1p = load_rnn(fsig_1p, fbkg_1p, sig_deco_1p,
                                       bkg_deco_1p)
y_3p, y_true_3p, w_3p, v_3p = load_rnn(fsig_3p, fbkg_3p, sig_deco_3p,
                                       bkg_deco_3p)

def corr(arr, y, w):
    L = []
    for i in range(10):
        a = arr[:, i]
        mask = ~np.isnan(a)
        r = pearsonr(a[mask], y[mask], weights=w[mask])
        L.append(r)

    return np.array(L)

is_sig_1p = y_true_1p == 1
is_sig_3p = y_true_3p == 1

c_pt_1p = corr(np.log10(v_1p["pttrack"]), y_1p, w_1p)
c_pt_3p = corr(np.log10(v_3p["pttrack"]), y_3p, w_3p)

c_ptsig_1p = corr(np.log10(v_1p["pttrack"][is_sig_1p]), y_1p[is_sig_1p], w_1p[is_sig_1p])
c_ptsig_3p = corr(np.log10(v_3p["pttrack"][is_sig_3p]), y_3p[is_sig_3p], w_3p[is_sig_3p])

c_ptbkg_1p = corr(np.log10(v_1p["pttrack"][~is_sig_1p]), y_1p[~is_sig_1p], w_1p[~is_sig_1p])
c_ptbkg_3p = corr(np.log10(v_3p["pttrack"][~is_sig_3p]), y_3p[~is_sig_3p], w_3p[~is_sig_3p])


c_d0_1p = corr(np.log10(np.abs(v_1p["d0"]) + 1e-6), y_1p, w_1p)
c_d0_3p = corr(np.log10(np.abs(v_3p["d0"]) + 1e-6), y_3p, w_3p)


c_z0_1p = corr(np.log10(np.abs(v_1p["z0"]) + 1e-6), y_1p, w_1p)
c_z0_3p = corr(np.log10(np.abs(v_3p["z0"]) + 1e-6), y_3p, w_3p)

c_z0sig_1p = corr(np.log10(np.abs(v_1p["z0"][is_sig_1p]) + 1e-6), y_1p[is_sig_1p], w_1p[is_sig_1p])
c_z0sig_3p = corr(np.log10(np.abs(v_3p["z0"][is_sig_3p]) + 1e-6), y_3p[is_sig_3p], w_3p[is_sig_3p])

c_z0bkg_1p = corr(np.log10(np.abs(v_1p["z0"][~is_sig_1p]) + 1e-6), y_1p[~is_sig_1p], w_1p[~is_sig_1p])
c_z0bkg_3p = corr(np.log10(np.abs(v_3p["z0"][~is_sig_3p]) + 1e-6), y_3p[~is_sig_3p], w_3p[~is_sig_3p])


c_dR_1p = corr(v_1p["dR"], y_1p, w_1p)
c_dR_3p = corr(v_3p["dR"], y_3p, w_3p)

c_pix_1p = corr(v_1p["pix"], y_1p, w_1p)
c_pix_3p = corr(v_3p["pix"], y_3p, w_3p)

c_pixsig_1p = corr(v_1p["pix"][is_sig_1p], y_1p[is_sig_1p], w_1p[is_sig_1p])
c_pixsig_3p = corr(v_3p["pix"][is_sig_3p], y_3p[is_sig_3p], w_3p[is_sig_3p])

c_pixbkg_1p = corr(v_1p["pix"][~is_sig_1p], y_1p[~is_sig_1p], w_1p[~is_sig_1p])
c_pixbkg_3p = corr(v_3p["pix"][~is_sig_3p], y_3p[~is_sig_3p], w_3p[~is_sig_3p])


print("log(pt):")
print(c_pt_1p)
print(c_pt_3p)

print("abs(d0+e):")
print(c_d0_1p)
print(c_d0_3p)

print("abs(z0+e):")
print(c_z0_1p)
print(c_z0_3p)

print("dR:")
print(c_dR_1p)
print(c_dR_3p)

print("Pix:")
print(c_pix_1p)
print(c_pix_3p)

print("Pixsig:")
print(c_pixsig_1p)
print(c_pixsig_3p)

print("Pixbkg:")
print(c_pixbkg_1p)
print(c_pixbkg_3p)


x = np.arange(1, 11)

fig, ax = plt.subplots()
ax.plot(x, c_pt_1p, "o", c="r", label="1-prong")
ax.plot(x, c_pt_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho( \log(p_\mathrm{T}^\mathrm{track}), p_\mathrm{RNN})$", ha="right", y=1.0)
ax.legend()
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("pt_corr.pdf")


fig, ax = plt.subplots()
ax.plot(x, c_ptsig_1p, "o", c="r", label="1-prong")
ax.plot(x, c_ptsig_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho( \log(p_\mathrm{T}^\mathrm{track}), p_\mathrm{RNN})$", ha="right", y=1.0)
ax.legend()
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("ptsig_corr.pdf")


fig, ax = plt.subplots()
ax.plot(x, c_ptbkg_1p, "o", c="r", label="1-prong")
ax.plot(x, c_ptbkg_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho( \log(p_\mathrm{T}^\mathrm{track}), p_\mathrm{RNN})$", ha="right", y=1.0)
ax.legend()
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("ptbkg_corr.pdf")




fig, ax = plt.subplots()
ax.plot(x, c_d0_1p, "o", c="r", label="1-prong")
ax.plot(x, c_d0_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left( \log\left(|d_0|\right), p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend()
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("d0_corr.pdf")

fig, ax = plt.subplots()
ax.plot(x, c_z0_1p, "o", c="r", label="1-prong")
ax.plot(x, c_z0_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left( \log\left(|z_0 \sin\theta| \right), p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("z0_corr.pdf")


fig, ax = plt.subplots()
ax.plot(x, c_z0sig_1p, "o", c="r", label="1-prong")
ax.plot(x, c_z0sig_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left( \log\left(|z_0 \sin\theta| \right), p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("z0sig_corr.pdf")

fig, ax = plt.subplots()
ax.plot(x, c_z0bkg_1p, "o", c="r", label="1-prong")
ax.plot(x, c_z0bkg_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left( \log\left(|z_0 \sin\theta| \right), p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("z0bkg_corr.pdf")


fig, ax = plt.subplots()
ax.plot(x, c_dR_1p, "o", c="r", label="1-prong")
ax.plot(x, c_dR_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left(\Delta R, p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("dR_corr.pdf")

fig, ax = plt.subplots()
ax.plot(x, c_pix_1p, "o", c="r", label="1-prong")
ax.plot(x, c_pix_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left(N_\mathrm{pixel}, p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("pix_corr.pdf")

fig, ax = plt.subplots()
ax.plot(x, c_pixsig_1p, "o", c="r", label="1-prong")
ax.plot(x, c_pixsig_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left(N_\mathrm{pixel}, p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("pixsig_corr.pdf")

fig, ax = plt.subplots()
ax.plot(x, c_pixbkg_1p, "o", c="r", label="1-prong")
ax.plot(x, c_pixbkg_3p, "o", c="b", label="3-prong")
ax.axhline(0.0, linestyle=":", linewidth=0.6, color="grey", zorder=0)
ax.set_xlim(0.5, 10.5)
ax.set_xticks(x)
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Track in sequence", ha="right", x=1.0)
ax.set_ylabel(r"$\rho\left(N_\mathrm{pixel}, p_\mathrm{RNN}\right)$", ha="right", y=1.0)
ax.legend(loc="lower right")
ax.tick_params(axis="x", which="minor", bottom="off", top="off")
fig.savefig("pixbkg_corr.pdf")
