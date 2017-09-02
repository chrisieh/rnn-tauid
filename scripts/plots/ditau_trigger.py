import argparse
from copy import copy

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.metrics import roc_curve

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48, pad_bottom=0.22)

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width

def roc(*args, **kwargs):
    fpr, tpr, thr = roc_curve(*args, **kwargs)
    nonzero = fpr != 0
    eff = tpr[nonzero]
    rej = 1.0 / fpr[nonzero]

    return eff, rej


def main(args):
    # Load signal
    with h5py.File(args.sig, "r", driver="family", memb_size=10*1024**3) as fin:
        s_eventIndex = fin["TauJets/eventIndex"][...]
        s_pt = fin["TauJets/pt"][...]
        s_eta = fin["TauJets/eta"][...]
        s_rnnScore = fin["TauJets/RNNScore"][...]
        s_truthPtVis = fin["TauJets/truthPtVis"][...]

        mask = (np.abs(s_eta) < 2.5) & ((np.abs(s_eta) < 1.0) | (np.abs(s_eta) > 1.4))
        s_eventIndex = s_eventIndex[mask]
        s_pt = s_pt[mask]
        s_eta = s_eta[mask]
        s_rnnScore = s_rnnScore[mask]
        s_truthPtVis = s_truthPtVis[mask]

    # Load background
    with h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as fin:
        b_eventIndex = fin["TauJets/eventIndex"][...]
        b_pt = fin["TauJets/pt"][...]
        b_eta = fin["TauJets/eta"][...]
        b_rnnScore = fin["TauJets/RNNScore"][...]

        mask = (np.abs(b_eta) < 2.5) & ((np.abs(b_eta) < 1.0) | (np.abs(b_eta) > 1.4))
        b_eventIndex = b_eventIndex[mask]
        b_pt = b_pt[mask]
        b_eta = b_eta[mask]
        b_rnnScore = b_rnnScore[mask]


    # Plot pt-spectrum
    fig, ax = plt.subplots()
    ax.hist(s_pt/1000.0, bins=10, range=(20, 60), normed=True,
            histtype="step", label="Signal")
    ax.hist(b_pt/1000.0, bins=10, range=(20, 60), normed=True,
            histtype="step", label="Background")
    ax.legend()
    ax.set_xlabel("Reco. tau pt / GeV")
    ax.set_ylabel("Normalised number of events")

    fig.savefig("pt.pdf")

    # Only ditau events
    unique, counts = np.unique(s_eventIndex, return_counts=True)
    ditau_idx = set(unique[counts == 2])

    ditaus = {}
    for i, idx in tqdm(enumerate(s_eventIndex), desc="Signal ditaus"):
        if idx not in ditau_idx:
            continue

        if idx in ditaus:
            pt1, pt2, s1, s2, pt1truth, pt2truth, eta1, eta2 = ditaus[idx]

            pt2 = s_pt[i]
            s2 = s_rnnScore[i]
            pt2truth = s_truthPtVis[i]
            eta2 = s_eta[i]

            # Order by rnn score
            if s1 < s2:
                pt1, pt2 = pt2, pt1
                s1, s2 = s2, s1
                pt1truth, pt2truth = pt2truth, pt1truth
                eta1, eta2 = eta2, eta1

            ditaus[idx] = (pt1, pt2, s1, s2, pt1truth, pt2truth, eta1, eta2)
        else:
            ditaus[idx] = (s_pt[i], 0.0, s_rnnScore[i], 0.0, s_truthPtVis[i], 0.0, s_eta[i], 0.0)

    # Jet events
    unique, counts = np.unique(b_eventIndex, return_counts=True)
    # At least two tau candidates
    ditau_jet_idx = set(unique[counts > 1])

    # Only take the leading two candidates in RNNScore
    ditaus_jet = {}
    for i, idx in tqdm(enumerate(b_eventIndex), desc="Background ditaus"):
        if idx not in ditau_jet_idx:
            continue

        if idx in ditaus_jet:
            pt1, pt2, s1, s2, eta1, eta2 = ditaus_jet[idx]

            pt2 = b_pt[i]
            s2 = b_rnnScore[i]
            eta2 = b_eta[i]

            if s1 < s2:
                pt1, pt2 = pt2, pt1
                s1, s2 = s2, s1
                eta1, eta2 = eta2, eta1

            ditaus_jet[idx] = (pt1, pt2, s1, s2, eta1, eta2)
        else:
            ditaus_jet[idx] = (b_pt[i], 0.0, b_rnnScore[i], 0.0, b_eta[i], 0.0)

    # Rate of pt selection
    # 33000.*(1.+0.4/2.5)/999500
    arr = np.array(ditaus_jet.values())
    weight = 31600.0 * (1.0 + 0.4 / 2.5) / 999500.0 # kHz

    leadpt = arr[:, 0]/1000.0
    subpt = arr[:, 1]/1000.0
    leadrnn = arr[:, 2]
    subrnn = arr[:, 3]

    # Selection RNN score > -0.013*pt[GeV]+1.04
    pass_id = (leadrnn > -0.013*leadpt + 1.04) & (subrnn > -0.013*subpt + 1.04)
    leadpt = leadpt[pass_id]
    subpt = subpt[pass_id]

    leadbins = np.linspace(25, 40, 41)
    subbins = np.linspace(25, 40, 41)

    row = []
    for lead in tqdm(leadbins):
        col = []
        for sub in subbins:
            if sub > lead:
                col.append(np.nan)
                continue

            # Else calculate
            col.append(weight * np.count_nonzero((leadpt > lead) & (subpt > sub)))

        row.append(col)

    pt_rate = np.array(row)
    pt_rate[np.isnan(pt_rate)] = -1111.

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(leadbins, subbins)
    im = ax.pcolormesh(xx, yy, pt_rate.T, vmin=0, cmap="plasma")
    im.cmap.set_under("white", alpha=0.0)
    cb = fig.colorbar(im)
    #cb.ax.minorticks_off()
    cb.ax.tick_params(length=4.0)
    cb.set_label("Rate / kHz", ha="right", y=1.0)

    ax.set_xlabel("Leading tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
    ax.set_ylabel("Subleading tau $p_\mathrm{T}$ / GeV", ha="right", y=1)
    fig.savefig("pt_rate.pdf")

    # Single tau efficiency
    pass_id = (s_rnnScore > -0.013 * s_pt / 1000.0 + 1.04)
    pt_bins = np.linspace(20, 200, 40)
    eff = binned_efficiency(s_truthPtVis/ 1000.0, pass_id, bins=pt_bins)

    fig, ax = plt.subplots()
    ax.errorbar(bin_center(pt_bins), eff.mean,
                xerr=bin_width(pt_bins) / 2.0, yerr=eff.std,
                fmt="o")
    ax.set_xlabel("pt / GeV")
    ax.set_ylabel("Efficiency")
    fig.savefig("taueff_single.pdf")


    # Tau efficiency
    arr = np.array(ditaus.values())

    leadpt = arr[:, 0]/1000.0
    subpt = arr[:, 1]/1000.0
    leadrnn = arr[:, 2]
    subrnn = arr[:, 3]
    leadpttruth = arr[:, 4] / 1000.0

    #pass_id = (leadrnn > 0.0) & (subrnn > 0.0)
    pass_id = (leadrnn > -0.013*leadpt + 1.04)# & (subrnn > -0.013*subpt + 1.04)


    pt_bins = np.linspace(20, 100, 15)
    eff = binned_efficiency(leadpttruth, pass_id, bins=pt_bins)

    fig, ax = plt.subplots()
    ax.errorbar(bin_center(pt_bins), eff.mean,
                xerr=bin_width(pt_bins) / 2.0, yerr=eff.std,
                fmt="o")
    ax.set_xlabel("pt / GeV")
    ax.set_ylabel("Efficiency")
    fig.savefig("taueff.pdf")


    # Plot probability product
    sig_pprod = []
    for k, v in ditaus.iteritems():
        pt1, pt2, s1, s2, pt1truth, pt2truth, eta1, eta2 = v
        sig_pprod.append(s1 * s2)

    bkg_pprod = []
    for k, v in ditaus_jet.iteritems():
        pt1, pt2, s1, s2, eta1, eta2 = v
        bkg_pprod.append(s1 * s2)

    sig_pprod = np.array(sig_pprod)
    bkg_pprod = np.array(bkg_pprod)

    fig, ax = plt.subplots()
    ax.hist(sig_pprod, bins=20, range=(0, 1), normed=True,
            histtype="step", label="Signal")
    ax.hist(bkg_pprod, bins=20, range=(0, 1), normed=True,
            histtype="step", label="Background")
    ax.legend()
    ax.set_xlabel("Probability product")
    ax.set_ylabel("Normalised number of events")

    fig.savefig("scoreprod.pdf")

    # ROC curve on probability product
    score = np.concatenate([sig_pprod, bkg_pprod])
    labels = np.concatenate([np.ones_like(sig_pprod), np.zeros_like(bkg_pprod)])

    eff, rej = roc(labels, score)


    fig, ax = plt.subplots()
    ax.plot(eff, rej, lw=0.8, c=(0.0, 0.0, 1.0))
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0.8, 1.0)
    ax.set_ylim(1, 10)

    fig.savefig("roc.pdf")

    # logistic regression
    s_arr = np.array(ditaus.values())[:, :4]
    b_arr = np.array(ditaus_jet.values())[:, :4]

    X = np.vstack([s_arr, b_arr])
    leadfrac = X[:, 0] / (X[:, 0] + X[:, 1])

    leadpt = X[:, 0] / 1000.0
    mu = leadpt.mean()
    std = leadpt.std()
    leadpt -= mu
    leadpt /= std

    subpt = X[:, 0] / 1000.0
    mu2 = subpt.mean()
    std2 = subpt.std()
    subpt -= mu2
    subpt /= std2

    X = np.hstack([leadpt[:, np.newaxis], subpt[:, np.newaxis], leadfrac[:, np.newaxis], X[:, 2:4],(X[:, 2] * X[:, 3])[:, np.newaxis]])
    Y = np.concatenate([np.ones(len(s_arr)), np.zeros(len(b_arr))])

    parameters = [
        {
            "penalty": ["none"],
            "fit_intercept": [False, True]
        }
    ]

    parameters = [
        {
            "penalty": ["none"],
            "fit_intercept": [False, True]
        },
        {
            "penalty": ["l1"],
            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
            "fit_intercept": [False, True]
        }
    ]

    # was 12345 then 123456
    np.random.seed(1234567)
    sgd = SGDClassifier(loss="log", class_weight="balanced", n_iter=50)
    gscv = GridSearchCV(sgd, parameters, n_jobs=2, scoring="neg_log_loss", cv=4,
                        verbose=2)
    gscv = gscv.fit(X, Y)

    print(gscv.best_params_)
    print(gscv.best_score_)

    scores_reg = gscv.predict_proba(X)[:, 1]

    # ROC curve on probability product
    eff, rej = roc(Y, scores_reg)

    fig, ax = plt.subplots()
    ax.plot(eff, rej)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0.8, 1.0)
    ax.set_ylim(1, 10)

    fig.savefig("roc_reg.pdf")

    # Tau efficiency after logistic regression
    arr = np.array(ditaus.values())

    leadpt = arr[:, 0] / 1000.0
    subpt = arr[:, 1] / 1000.0
    leadrnn = arr[:, 2]
    subrnn = arr[:, 3]
    leadpttruth = arr[:, 4] / 1000.0
    subpttruth = arr[:, 5] / 1000.0
    logreg = scores_reg[Y==1]

    truth_mask = (leadpttruth > 25) & (subpttruth > 25)
    pass_id = (logreg > 0.925) & (leadpt > 25) & (subpt > 25)
    pass_id = pass_id[truth_mask]

    # pt_bins = np.linspace(25, 120, 18)
    pt_bins = np.logspace(np.log10(25), np.log10(120), 14)

    # Bootstrap loop
    results = []
    stds = []
    n_bootstrap = 1000
    for i in tqdm(range(n_bootstrap)):
        # Resample
        length = len(leadpttruth[truth_mask])
        assert length == len(pass_id)

        idx = np.random.randint(length, size=length)
        eff = binned_efficiency(leadpttruth[truth_mask][idx], pass_id[idx],
                                bins=pt_bins)
        results.append(eff.mean)
        stds.append(eff.std)

    results = np.array(results)
    stds = np.array(stds).mean(axis=0)

    sigma = 0.682689492137
    one_minus_sigma = 1.0 - sigma

    ymedian = np.nanpercentile(results, 50., axis=0)
    maxeff = ymedian == 1.0
    ylow = ymedian - np.nanpercentile(results, 100. * one_minus_sigma / 2.0, axis=0)
    ylow[maxeff] = 2*stds[maxeff]
    yhigh = np.nanpercentile(results, 100. - 100. * one_minus_sigma / 2.0, axis=0) - ymedian

    yerr = np.vstack([ylow[np.newaxis, :], yhigh[np.newaxis, :]])

    fig, ax = plt.subplots()
    ax.errorbar(bin_center(pt_bins), ymedian,
                xerr=bin_width(pt_bins) / 2.0, yerr=yerr,
                fmt="o", color=(0.0, 0.0, 1.0))
    ax.set_xlabel("Leading tau $p_\mathrm{T}^\mathrm{true}$ / GeV", ha="right", x=1)
    ax.set_ylabel("Di-tau efficiency", ha="right", y=1)
    ax.set_ylim(0, 1.1)
    fig.savefig("taueff_reg.pdf")


    # Rate of pt selection
    # 33000.*(1.+0.4/2.5)/999500
    arr = np.array(ditaus_jet.values())
    weight = 31600.0 * (1.0 + 0.4 / 2.5) / 999500.0 # kHz

    leadpt = arr[:, 0]/1000.0
    subpt = arr[:, 1]/1000.0
    leadrnn = arr[:, 2]
    subrnn = arr[:, 3]
    logreg = scores_reg[Y==0]

    # Selection RNN score > -0.013*pt[GeV]+1.04
    pass_id = logreg > 0.925
    leadpt = leadpt[pass_id]
    subpt = subpt[pass_id]

    leadbins = np.linspace(20, 40, 161)
    subbins = np.linspace(20, 40, 161)

    row = []
    for lead in tqdm(leadbins):
        col = []
        for sub in subbins:
            if sub > lead:
                col.append(np.nan)
                continue

            # Else calculate
            col.append(weight * np.count_nonzero((leadpt > lead) & (subpt > sub)))

        row.append(col)

    pt_rate = np.array(row)
    pt_rate[np.isnan(pt_rate)] = -1111.

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(leadbins, subbins)
    im = ax.pcolormesh(xx, yy, pt_rate.T, vmin=0, cmap="plasma")
    cont = ax.contour(xx, yy, pt_rate.T, [200], colors="white", alpha=0.0)
    cont_points = cont.allsegs[0][0]
    cont_x = cont_points[:, 0]
    cont_y = cont_points[:, 1]
    maxidx = np.argmax(cont_y)+1

    ax.plot(cont_x[:maxidx], cont_y[:maxidx], label="200 kHz", c=(0.0, 0.0, 1.0))
    ax.legend(loc="upper left")

    im.cmap.set_under("white", alpha=0.0)
    cb = fig.colorbar(im)
    #cb.ax.minorticks_off()
    cb.ax.tick_params(length=4.0)
    cb.ax.tick_params(which="minor", length=2.0)
    cb.set_label("Rate / kHz", ha="right", y=1.0)
    ax.set_yticks([20, 25, 30, 35 ,40])
    ax.set_xticks([20, 25, 30, 35, 40])

    ax.set_xlabel("Leading tau $p_\mathrm{T}$ / GeV", ha="right", x=1)
    ax.set_ylabel("Subleading tau $p_\mathrm{T}$ / GeV", ha="right", y=1)
    fig.savefig("pt_rate_reg.pdf")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")

    args = parser.parse_args()
    main(args)
