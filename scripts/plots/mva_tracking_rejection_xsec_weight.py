import os
import numpy as np

from root_numpy import root2array
from tqdm import tqdm

import matplotlib as mpl
mpl.use("PDF")
import matplotlib.pyplot as plt
from rnn_tauid.common.mpl_setup import mpl_setup
mpl_setup(scale=0.48)

from rnn_tauid.evaluation.misc import binned_efficiency, bin_center, bin_width

# My samples:
# JZ1W: nevent = 2020000 xsec = 78420000
# JZ2W: nevent = 1994000 xsec = 2433200
# JZ3W: nevent = 7801500 xsec = 26454
# JZ4W: nevent = 7973500 xsec = 254.63
# JZ5W: nevent = 7948500 xsec = 4.5535
# JZ6W: nevent = 1981000 xsec = 0.25753

prefix = "/lustre/atlas/group/higgs/cdeutsch/StreamTriggerIDDev_flat/v02"

nevent_xsec = {
    "JZ1W": (2020000, 78420000),
    "JZ2W": (1994000, 2433200),
    "JZ3W": (7801500, 26454),
    "JZ4W": (7973500, 254.63),
    "JZ5W": (7948500, 4.5535),
    "JZ6W": (1981000, 0.25753)
}
br = ["TauJets.pt", "TauJets.eta", "TauJets.nTracks"]
sel = "(TauJets.pt > 20000) && (TMath::Abs(TauJets.eta) < 1.37 || TMath::Abs(TauJets.eta) > 1.52)"

samples = {}
weights = {}
sample_names = ["JZ{}W".format(i) for i in range(1, 7)]
for sample_name in tqdm(sample_names, desc="Loading samples"):
    path = os.path.join(prefix, sample_name, "*.root")
    samples[sample_name] = root2array(path, treename="CollectionTree",
                                      branches=br, selection=sel)

    # Event weights
    size = len(samples[sample_name])
    nevent, xsec = nevent_xsec[sample_name]
    weights[sample_name] = np.ones(size) * xsec / nevent

pt = np.concatenate([samples[sample_name]["TauJets.pt"] for sample_name in
                     sample_names]) / 1000.0
nTracks = np.concatenate([samples[sample_name]["TauJets.nTracks"] for
                          sample_name in sample_names])
weight = np.concatenate([weights[sample_name] for sample_name in sample_names])

oneprong = (nTracks == 1)
threeprong = (nTracks == 3)

# Setup
bins = np.linspace(20, 400, 25)
bin_midpoint = bin_center(bins)
bin_half_width = bin_width(bins) / 2.0

# 1-prong
pass_1p = np.histogram(pt[oneprong], bins=bins, weights=weight[oneprong])
total_1p = np.histogram(pt, bins=bins, weights=weight)
rej_1p = total_1p[0] / pass_1p[0]


# 3-prong
pass_3p = np.histogram(pt[threeprong], bins=bins, weights=weight[threeprong])
total_3p = np.histogram(pt, bins=bins, weights=weight)
rej_3p = total_3p[0] / pass_3p[0]


fig, ax = plt.subplots()
ax.errorbar(bin_midpoint, rej_1p, xerr=bin_half_width, fmt="o", c="r",
            label="1-track")
ax.errorbar(bin_midpoint, rej_3p, xerr=bin_half_width, fmt="o", c="b",
            label="3-track")
ax.set_xlabel(r"Reconstructed tau $p_\mathrm{T}$ / GeV", ha="right", x=1.0)
ax.set_ylabel("Rejection", ha="right", y=1.0)

ylim = ax.get_ylim()
ax.set_ylim(0, 65)

ax.legend()

fig.savefig("mva_tracking_rejection.pdf")
