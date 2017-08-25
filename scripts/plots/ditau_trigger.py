import argparse

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
mpl_setup(scale=0.48)


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
        s_rnnScore = fin["TauJets/RNNScore"][...]

    # Load background
    with h5py.File(args.bkg, "r", driver="family", memb_size=10*1024**3) as fin:
        b_eventIndex = fin["TauJets/eventIndex"][...]
        b_pt = fin["TauJets/pt"][...]
        b_rnnScore = fin["TauJets/RNNScore"][...]

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
            pt1, pt2, s1, s2 = ditaus[idx]

            pt2 = s_pt[i]
            s2 = s_rnnScore[i]

            # Order by rnn score
            if s1 < s2:
                pt1, pt2 = pt2, pt1
                s1, s2 = s2, s1

            ditaus[idx] = (pt1, pt2, s1, s2)
        else:
            ditaus[idx] = (s_pt[i], 0.0, s_rnnScore[i], 0.0)

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
            pt1, pt2, s1, s2 = ditaus_jet[idx]

            pt, s = b_pt[i], b_rnnScore[i]
            if s > s2:
                pt2, s2 = pt, s

            if s1 < s2:
                pt1, pt2 = pt2, pt1
                s1, s2 = s2, s1

            ditaus_jet[idx] = (pt1, pt2, s1, s2)
        else:
            ditaus_jet[idx] = (b_pt[i], 0.0, b_rnnScore[i], 0.0)

    # Plot probability product
    sig_pprod = []
    for k, v in ditaus.iteritems():
        pt1, pt2, s1, s2 = v
        sig_pprod.append(s1 * s2)

    bkg_pprod = []
    for k, v in ditaus_jet.iteritems():
        pt1, pt2, s1, s2 = v
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
    ax.plot(eff, rej)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0.8, 1.0)
    ax.set_ylim(1, 10)

    fig.savefig("roc.pdf")

    # logistic regression
    s_arr = np.array(ditaus.values())[:, 2:]
    b_arr = np.array(ditaus_jet.values())[:, 2:]

    X = np.vstack([s_arr, b_arr])
    X = np.hstack([X, (X[:, 0] * X[:, 1])[:, np.newaxis]])
    Y = np.concatenate([np.ones(len(s_arr)), np.zeros(len(b_arr))])

    parameters = [
        {
            "penalty": ["none"],
            "fit_intercept": [False, True]
        },
        {
            "penalty": ["l1", "l2"],
            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
            "fit_intercept": [False, True]
        }
    ]
    sgd = SGDClassifier(loss="log", class_weight="balanced", n_iter=20)
    gscv = GridSearchCV(sgd, parameters, n_jobs=4, scoring="neg_log_loss", cv=5,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig")
    parser.add_argument("bkg")

    args = parser.parse_args()
    main(args)
