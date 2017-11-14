from array import array

import numpy as np
import h5py
from ROOT import TMVA
from root_numpy.tmva import evaluate_reader


# Samples
fsig_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig1P_v08_%d.h5"
fbkg_1p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg1P_v08_%d.h5"
fsig_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/sig3P_v08_%d.h5"
fbkg_3p = "/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_flat/bkg3P_v08_%d.h5"

# 1-prong
xml_1p = "/lustre/user/cdeutsch/bdt_tauid/variable_importance/1p/iter_1/" + \
         "ChPiEMEOverCaloEME.alg/model.xml"

var_1p = [
    "TMath::Min(TauJets.centFrac,1.0)",
    "TMath::Log10(TMath::Max(0.1,TauJets.etOverPtLeadTrk))",
    "TauJets.innerTrkAvgDist",
    "TMath::Min(TauJets.absipSigLeadTrk,30)",
    "TauJets.SumPtTrkFrac",
    "TMath::Log10(TMath::Max(0.01,TauJets.EMPOverTrkSysP))",
    "TMath::Min(TauJets.ptRatioEflowApprox,4)",
    "TMath::Log10(TMath::Max(140,TauJets.mEflowApprox))",
    "TauJets.ptIntermediateAxis > 100000.0 ? 100000.0 : TauJets.ptIntermediateAxis"
]

reader = TMVA.Reader()
for v in var_1p:
    reader.AddVariable(v, array("f", [0.]))
reader.BookMVA("BDT method", xml_1p)


def calc_1p(reader, fin):
    with h5py.File(fin, "r", driver="family", memb_size=10*1024**3) as f:
        size = len(f["TauJets/centFrac"])

        data = np.zeros((size, 9))

        data[:, 0] = np.minimum(f["TauJets/centFrac"][...], 1.0)
        data[:, 1] = np.log10(np.maximum(0.1, f["TauJets/etOverPtLeadTrk"][...]))
        data[:, 2] = f["TauJets/innerTrkAvgDist"][...]
        data[:, 3] = np.minimum(f["TauJets/absipSigLeadTrk"][...], 30.0)
        data[:, 4] = f["TauJets/SumPtTrkFrac"][...]
        data[:, 5] = np.log10(np.maximum(0.01, f["TauJets/EMPOverTrkSysP"][...]))
        data[:, 6] = np.minimum(f["TauJets/ptRatioEflowApprox"][...], 4.0)
        data[:, 7] = np.log10(np.maximum(140.0, f["TauJets/mEflowApprox"][...]))
        data[:, 8] = np.minimum(f["TauJets/ptIntermediateAxis"], 100000.0)

    return evaluate_reader(reader, "BDT method", data)


print("Decorating 1-prong signal...")
sig_bdt_1p = calc_1p(reader, fsig_1p)
print(sig_bdt_1p)

with h5py.File("sig1P_bdt.h5", "w") as f:
    f.create_dataset("score", data=sig_bdt_1p)

print("Decorating 1-prong background...")
bkg_bdt_1p = calc_1p(reader, fbkg_1p)
print(bkg_bdt_1p)

with h5py.File("bkg1P_bdt.h5", "w") as f:
    f.create_dataset("score", data=bkg_bdt_1p)


# 3-prong
xml_3p = "/lustre/user/cdeutsch/bdt_tauid/variable_importance/3p/iter_2/" + \
         "ChPiEMEOverCaloEME.alg/model.xml"

var_3p = [
    "TMath::Min(TauJets.centFrac,1.0)",
    "TMath::Log10(TMath::Max(0.1,TauJets.etOverPtLeadTrk))",
    "TauJets.dRmax",
    "TMath::Log10(TMath::Max(0.01,TauJets.trFlightPathSig))",
    "TMath::Log10(TMath::Max(140,TauJets.massTrkSys))",
    "TMath::Log10(TMath::Max(0.01,TauJets.EMPOverTrkSysP))",
    "TMath::Min(TauJets.ptRatioEflowApprox,4)",
    "TMath::Log10(TMath::Max(140,TauJets.mEflowApprox))",
    "TauJets.SumPtTrkFrac",
    "TauJets.ptIntermediateAxis > 100000.0 ? 100000.0 : TauJets.ptIntermediateAxis"
]

reader = TMVA.Reader()
for v in var_3p:
    reader.AddVariable(v, array("f", [0.]))
reader.BookMVA("BDT method", xml_3p)


def calc_3p(reader, fin):
    with h5py.File(fin, "r", driver="family", memb_size=10*1024**3) as f:
        size = len(f["TauJets/centFrac"])

        data = np.zeros((size, 10))

        data[:, 0] = np.minimum(f["TauJets/centFrac"][...], 1.0)
        data[:, 1] = np.log10(np.maximum(0.1, f["TauJets/etOverPtLeadTrk"][...]))
        data[:, 2] = f["TauJets/dRmax"][...]
        data[:, 3] = np.log10(np.maximum(0.01, f["TauJets/trFlightPathSig"][...]))
        data[:, 4] = np.log10(np.maximum(140.0, f["TauJets/massTrkSys"][...]))
        data[:, 5] = np.log10(np.maximum(0.01, f["TauJets/EMPOverTrkSysP"][...]))
        data[:, 6] = np.minimum(f["TauJets/ptRatioEflowApprox"][...], 4.0)
        data[:, 7] = np.log10(np.maximum(140.0, f["TauJets/mEflowApprox"][...]))
        data[:, 8] = f["TauJets/SumPtTrkFrac"][...]
        data[:, 9] = np.minimum(f["TauJets/ptIntermediateAxis"], 100000.0)

    return evaluate_reader(reader, "BDT method", data)


print("Decorating 3-prong signal...")
sig_bdt_3p = calc_3p(reader, fsig_3p)
print(sig_bdt_3p)

with h5py.File("sig3P_bdt.h5", "w") as f:
    f.create_dataset("score", data=sig_bdt_3p)

print("Decorating 3-prong background...")
bkg_bdt_3p = calc_3p(reader, fbkg_3p)
print(bkg_bdt_3p)

with h5py.File("bkg3P_bdt.h5", "w") as f:
    f.create_dataset("score", data=bkg_bdt_3p)
