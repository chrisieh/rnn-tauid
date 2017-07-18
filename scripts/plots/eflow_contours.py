import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from scipy.stats import gaussian_kde

from root_numpy import root2array


sigf = "sig1P_test.root"
bkgf = "bkg1P_weight_test.root"

tree = "CollectionTree"
weight = "weight"
branches = ["TauJets.ptRatioEflowApprox", "TauJets.mEflowApprox"]

sig = root2array(sigf, treename=tree, branches=branches, step=50)
bkg = root2array(bkgf, treename=tree, branches=branches, step=30)

print(len(sig))
print(len(bkg))

pt = sig["TauJets.ptRatioEflowApprox"]
m = sig["TauJets.mEflowApprox"] / 1000
pt2 = bkg["TauJets.ptRatioEflowApprox"]
m2 = bkg["TauJets.mEflowApprox"] / 1000

data = np.vstack([pt, m])
data2 = np.vstack([pt2, m2])

sig_kde = gaussian_kde(data, bw_method=0.002)
bkg_kde = gaussian_kde(data2, bw_method=0.01)

xx, yy = np.meshgrid(np.linspace(0.0, 2.0, 60), np.linspace(0.0, 5.0, 60))
pos = np.vstack([xx.ravel(), yy.ravel()])
f = np.reshape(sig_kde(pos).T, xx.shape)
f2 = np.reshape(bkg_kde(pos).T, xx.shape)

print(np.percentile(f, np.arange(20, 100, 20)))

fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 3)
ax.set_xlabel("ptRatioEflowApprox", ha="right", x=1)
ax.set_ylabel("mEflowApprox", ha="right", y=1)

contour = ax.contour(xx, yy, f, [0.2, 0.6, 1.0, 1.4], colors="r")
contour2 = ax.contour(xx, yy, f2, [0.2, 0.6, 1.0, 1.4], colors="b")

ax.clabel(contour, inline=1, fontsize=6)
ax.clabel(contour2, inline=1, fontsize=6)

fig.savefig("contour.pdf")
