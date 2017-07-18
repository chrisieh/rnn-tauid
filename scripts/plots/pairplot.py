import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("PDF")

import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from matplotlib.colors import Normalize, LogNorm
import seaborn as sns

from root_numpy import root2array


sigf = "sig1P_test.root"
bkgf = "bkg1P_weight_test.root"

tree = "CollectionTree"
weight = "weight"
branches = ["TauJets.centFrac", "TauJets.etOverPtLeadTrk",
            "TauJets.innerTrkAvgDist", "TauJets.absipSigLeadTrk",
            "TauJets.SumPtTrkFrac", "TauJets.ChPiEMEOverCaloEME",
            "TauJets.EMPOverTrkSysP", "TauJets.ptRatioEflowApprox",
            "TauJets.mEflowApprox"]

sig = root2array(sigf, treename=tree, branches=branches, step=5000)
bkg = root2array(bkgf, treename=tree, branches=branches, step=3000)

print(len(sig))
print(len(bkg))

df1 = pd.DataFrame(sig)
df1["class"] = 1
df2 = pd.DataFrame(bkg)
df2["class"] = 0

df = pd.concat([df1, df2])
del df1, df2

df["TauJets.etOverPtLeadTrk"] = np.clip(df["TauJets.etOverPtLeadTrk"], 0, 20)
df["TauJets.absipSigLeadTrk"] = np.clip(df["TauJets.absipSigLeadTrk"], 0, 20)
df["TauJets.ChPiEMEOverCaloEME"] = np.clip(df["TauJets.ChPiEMEOverCaloEME"], -2, 2)
df["TauJets.EMPOverTrkSysP"] = np.clip(df["TauJets.EMPOverTrkSysP"], 0, 25)
df["TauJets.ptRatioEflowApprox"] = np.clip(df["TauJets.ptRatioEflowApprox"], 0, 2.5)
df["TauJets.mEflowApprox"] = np.clip(df["TauJets.mEflowApprox"]/1000.0, 0, 10)

g = sns.pairplot(df, hue="class", markers=",", vars=branches)
g.savefig("pairplot.pdf")
