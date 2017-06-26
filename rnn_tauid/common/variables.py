from functools import partial

import numpy as np
from rnn_tauid.common.preprocessing import scale, scale_flat, robust_scale, \
                                           constant_scale


# Template for log10(x + epsilon)
def log10_epsilon(datafile, dest, source_sel=None, dest_sel=None, var=None,
                  epsilon=None):
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    if epsilon:
        np.add(dest[dest_sel], epsilon, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


# Template for log10(abs(x) + epsilon)
def abs_log10_epsilon(datafile, dest, source_sel=None, dest_sel=None, var=None,
                      epsilon=None):
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.abs(dest[dest_sel], out=dest[dest_sel])
    if epsilon:
        np.add(dest[dest_sel], epsilon, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def abs_var(datafile, dest, source_sel=None, dest_sel=None, var=None):
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.abs(dest[dest_sel], out=dest[dest_sel])


# Track variables
pt_log = partial(log10_epsilon, var="TauTracks/pt")

d0_abs = partial(abs_var, var="TauTracks/d0")

d0_abs_log = partial(abs_log10_epsilon,  var="TauTracks/d0", epsilon=1e-6)

z0sinThetaTJVA_abs = partial(abs_var, var="TauTracks/z0sinThetaTJVA")

z0sinThetaTJVA_abs_log = partial(abs_log10_epsilon,
                                 var="TauTracks/z0sinThetaTJVA", epsilon=1e-6)

rConv = partial(abs_var, var="TauTracks/rConvII")


def pt_frac(datafile, dest, source_sel=None, dest_sel=None):
    pt_track = datafile["TauTracks/pt"][source_sel]

    datafile["TauTracks/pt"].read_direct(dest, source_sel=source_sel,
                                         dest_sel=dest_sel)

    pt_jetseed = datafile["TauJets/ptJetSeed"][source_sel[0]]
    pt_jetseed = pt_jetseed[:, np.newaxis]

    dest[dest_sel] = pt_track / pt_jetseed


def pt_asym(datafile, dest, source_sel=None, dest_sel=None):
    pt_track = datafile["TauTracks/pt"][source_sel]

    datafile["TauTracks/pt"].read_direct(dest, source_sel=source_sel,
                                         dest_sel=dest_sel)

    pt_jetseed = datafile["TauJets/ptJetSeed"][source_sel[0]]
    pt_jetseed = pt_jetseed[:, np.newaxis]

    dest[dest_sel] = (pt_track - pt_jetseed) / (pt_track + pt_jetseed)


def pt_jetseed_log(datafile, dest, source_sel=None, dest_sel=None):
    pt = datafile["TauJets/ptJetSeed"][source_sel[0]]
    dest[dest_sel] = np.log10(pt[:, np.newaxis])


# Cluster variables
et_log = partial(
    log10_epsilon, var="TauClusters/et")

SECOND_R_log = partial(
    log10_epsilon, var="TauClusters/SECOND_R", epsilon=0.1)

SECOND_LAMBDA_log = partial(
    log10_epsilon, var="TauClusters/SECOND_LAMBDA", epsilon=0.1)

FIRST_ENG_DENS_log = partial(
    log10_epsilon, var="TauClusters/FIRST_ENG_DENS", epsilon=1e-6)

CENTER_LAMBDA_log = partial(
    log10_epsilon, var="TauClusters/CENTER_LAMBDA", epsilon=1e-6)

# Common ID variables
etOverPtLeadTrk_log = partial(
    log10_epsilon, var="TauJets/etOverPtLeadTrk", epsilon=1e-6)

absipSigLeadTrk_log = partial(
    log10_epsilon, var="TauJets/absipSigLeadTrk", epsilon=1e-6)

ptRatioEflowApprox_log = partial(
    log10_epsilon, var="TauJets/ptRatioEflowApprox", epsilon=1e-6)

mEflowAprox_log = partial(
    log10_epsilon, var="TauJets/mEflowApprox", epsilon=1e-6)

ptIntermediateAxis_log = partial(
    log10_epsilon, var="TauJets/ptIntermediateAxis")

mEflowApprox_log = partial(
    log10_epsilon, var="TauJets/mEflowApprox", epsilon=1e-6)

massTrkSys_log = partial(
    log10_epsilon, var="TauJets/massTrkSys", epsilon=1e-6)


def EMPOverTrkSysP_clip_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauJets/EMPOverTrkSysP"].read_direct(dest, source_sel=source_sel,
                                                   dest_sel=dest_sel)
    np.clip(dest[dest_sel], 1e-3, np.inf, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


# PFO variables
def Eta(datafile, dest, source_sel=None, dest_sel=None, var="TauPFOs/chargedEta"):
    # Hack to set nans
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])

    eta = datafile["TauJets/Eta"]
    np.add(dest[dest_sel], eta[source_sel[0]][:, np.newaxis], out=dest[dest_sel])


def Phi(datafile, dest, source_sel=None, dest_sel=None, var="TauPFOs/chargedPhi"):
    # Hack to set nans
    datafile[var].read_direct(dest, source_sel=source_sel,
                                                          dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])

    phi = datafile["TauJets/Phi"]
    np.add(dest[dest_sel], phi[source_sel[0]][:, np.newaxis], out=dest[dest_sel])


def dEta(datafile, dest, source_sel=None, dest_sel=None, var="TauPFOs/chargedEta"):
    eta_jet = datafile["TauJets/Eta"][source_sel[0]]
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.subtract(dest[dest_sel], eta_jet[:, np.newaxis], out=dest[dest_sel])


def dPhi(datafile, dest, source_sel=None, dest_sel=None, var="TauPFOs/chargedPhi"):
    phi_jet = datafile["TauJets/Phi"][source_sel[0]]
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.subtract(dest[dest_sel], phi_jet[:, np.newaxis], out=dest[dest_sel])
    np.add(dest[dest_sel], np.pi, out=dest[dest_sel])
    np.fmod(dest[dest_sel], 2 * np.pi, out=dest[dest_sel])
    np.subtract(dest[dest_sel], np.pi, out=dest[dest_sel])


def Pt_jet_log(datafile, dest, source_sel=None, dest_sel=None, var="TauPFOs/chargedPt"):
    # Hack to set nans
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])
    pt = datafile["TauJets/Pt"]

    np.add(dest[dest_sel], np.log10(pt[source_sel[0]])[:, np.newaxis],
           out=dest[dest_sel])


def PtSubRatio(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauPFOs/neutralPtSub"].read_direct(dest, source_sel=source_sel,
                                                 dest_sel=dest_sel)
    pt = datafile["TauPFOs/neutralPt"][source_sel]
    np.add(pt, dest[dest_sel], out=pt)
    np.divide(dest[dest_sel], pt, out=dest[dest_sel])

# dPhi, dEta for extrapolated conversion tracks

def dEta_extrap(datafile, dest, source_sel=None, dest_sel=None, var="TauConv/eta_extrap"):
    eta_jet = datafile["TauJets/Eta"][source_sel[0]]
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)

    # Set default value to nan
    dest[dest_sel][dest[dest_sel] == -10.0] = np.nan

    np.subtract(dest[dest_sel], eta_jet[:, np.newaxis], out=dest[dest_sel])


def dPhi_extrap(datafile, dest, source_sel=None, dest_sel=None, var="TauConv/phi_extrap"):
    phi_jet = datafile["TauJets/Phi"][source_sel[0]]
    datafile[var].read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)

    # Set default value to nan
    dest[dest_sel][dest[dest_sel] == -10.0] = np.nan

    np.subtract(dest[dest_sel], phi_jet[:, np.newaxis], out=dest[dest_sel])
    np.add(dest[dest_sel], np.pi, out=dest[dest_sel])
    np.fmod(dest[dest_sel], 2 * np.pi, out=dest[dest_sel])
    np.subtract(dest[dest_sel], np.pi, out=dest[dest_sel])


# Charged & neutral PFOs
charged_Eta = partial(Eta, var="TauPFOs/chargedEta")
neutral_Eta = partial(Eta, var="TauPFOs/neutralEta")
shot_Eta = partial(Eta, var="TauPFOs/shotEta")
hadronic_Eta = partial(Eta, var="TauPFOs/hadronicEta")
neutral_Eta_bdtsort = partial(Eta, var="TauPFOs/neutralEta_BDTSort")

charged_Phi = partial(Phi, var="TauPFOs/chargedPhi")
neutral_Phi = partial(Phi, var="TauPFOs/neutralPhi")
shot_Phi = partial(Phi, var="TauPFOs/shotPhi")
hadronic_Phi = partial(Phi, var="TauPFOs/hadronicPhi")
neutral_Phi_bdtsort = partial(Phi, var="TauPFOs/neutralPhi_BDTSort")

charged_dEta = partial(dEta, var="TauPFOs/chargedEta")
neutral_dEta = partial(dEta, var="TauPFOs/neutralEta")
shot_dEta = partial(dEta, var="TauPFOs/shotEta")
hadronic_dEta = partial(dEta, var="TauPFOs/hadronicEta")
neutral_dEta_bdtsort = partial(dEta, var="TauPFOs/neutralEta_BDTSort")

charged_dPhi = partial(dPhi, var="TauPFOs/chargedPhi")
neutral_dPhi = partial(dPhi, var="TauPFOs/neutralPhi")
shot_dPhi = partial(dPhi, var="TauPFOs/shotPhi")
hadronic_dPhi = partial(dPhi, var="TauPFOs/hadronicPhi")
neutral_dPhi_bdtsort = partial(dPhi, var="TauPFOs/neutralPhi_BDTSort")

charged_Pt_log = partial(log10_epsilon, var="TauPFOs/chargedPt")
neutral_Pt_log = partial(log10_epsilon, var="TauPFOs/neutralPt")
shot_Pt_log = partial(log10_epsilon, var="TauPFOs/shotPt")
hadronic_Pt_log = partial(log10_epsilon, var="TauPFOs/hadronicPt")
neutral_Pt_log_bdtsort = partial(log10_epsilon, var="TauPFOs/neutralPt_BDTSort")

charged_Pt_jet_log = partial(Pt_jet_log, var="TauPFOs/chargedPt")
neutral_Pt_jet_log = partial(Pt_jet_log, var="TauPFOs/neutralPt")
shot_Pt_jet_log = partial(Pt_jet_log, var="TauPFOs/shotPt")
hadronic_Pt_jet_log = partial(Pt_jet_log, var="TauPFOs/hadronicPt")
neutral_Pt_jet_log_bdtsort = partial(Pt_jet_log, var="TauPFOs/neutralPt_BDTSort")

# Moments
neutral_SECOND_R_log = partial(log10_epsilon, var="TauPFOs/neutral_SECOND_R", epsilon=1)
neutral_secondEtaWRTClusterPosition_EM1_log =partial(
    log10_epsilon, var="TauPFOs/neutral_secondEtaWRTClusterPosition_EM1", epsilon=1e-6)
neutral_SECOND_ENG_DENS_log = partial(log10_epsilon, var="TauPFOs/neutral_SECOND_ENG_DENS",
                                      epsilon=1e-8)


# Conversion tracks
conversion_Eta = partial(Eta, var="TauConv/eta")
conversion_Phi = partial(Phi, var="TauConv/phi")
conversion_dEta = partial(dEta, var="TauConv/eta")
conversion_dPhi = partial(dPhi, var="TauConv/phi")

conversion_dEta_extrapol = partial(dEta_extrap, var="TauConv/eta_extrap")
conversion_dPhi_extrapol = partial(dPhi_extrap, var="TauConv/phi_extrap")

conversion_Pt_log = partial(log10_epsilon, var="TauConv/pt")
conversion_Pt_jet_log = partial(Pt_jet_log, var="TauConv/pt")


track_vars = [
    ("TauTracks/pt_log", pt_log, scale),
    ("TauTracks/pt_asym", pt_asym, scale),
    ("TauTracks/d0_abs_log", d0_abs_log, scale),
    ("TauTracks/z0sinThetaTJVA_abs_log", z0sinThetaTJVA_abs_log, scale),
    ("TauTracks/dRJetSeedAxis", None, partial(constant_scale, scale=0.4)),
    ("TauTracks/eProbabilityHT", None, None),
    ("TauTracks/nInnermostPixelHits", None, partial(constant_scale, scale=3)),
    ("TauTracks/nPixelHits", None, partial(constant_scale, scale=11)),
    ("TauTracks/nSCTHits", None, partial(constant_scale, scale=20))
]

cluster_vars = [
    ("TauClusters/et_log", et_log, scale),
    ("TauClusters/psfrac", None, None),
    ("TauClusters/em1frac", None, None),
    ("TauClusters/em2frac", None, None),
    ("TauClusters/em3frac", None, None),
    ("TauClusters/dRJetSeedAxis", None, partial(constant_scale, scale=0.4)),
    ("TauClusters/EM_PROBABILITY", None, None),
    ("TauClusters/SECOND_R", SECOND_R_log, scale),
    ("TauClusters/SECOND_LAMBDA", SECOND_LAMBDA_log, scale),
    ("TauClusters/FIRST_ENG_DENS", FIRST_ENG_DENS_log, scale),
    ("TauClusters/CENTER_LAMBDA", CENTER_LAMBDA_log, scale),
    ("TauClusters/ENG_FRAC_MAX", None, None)
]

id1p_vars = [
    ("TauJets/centFrac", None, scale_flat),
    ("TauJets/innerTrkAvgDist", None, scale_flat),
    ("TauJets/SumPtTrkFrac", None, scale_flat),
    ("TauJets/etOverPtLeadTrk", etOverPtLeadTrk_log, scale_flat),
    ("TauJets/absipSigLeadTrk", absipSigLeadTrk_log, scale_flat),
    ("TauJets/EMPOverTrkSysP", EMPOverTrkSysP_clip_log, scale_flat),
    ("TauJets/ptRatioEflowApprox", ptRatioEflowApprox_log, scale_flat),
    ("TauJets/mEflowApprox", mEflowApprox_log, scale_flat),
    ("TauJets/ptIntermediateAxis", ptIntermediateAxis_log, scale_flat)
]

id3p_vars = [
    ("TauJets/centFrac", None, scale_flat),
    ("TauJets/innerTrkAvgDist", None, scale_flat),
    ("TauJets/SumPtTrkFrac", None, scale_flat),
    ("TauJets/trFlightPathSig", None, scale_flat),
    ("TauJets/dRmax", None, scale_flat),
    ("TauJets/massTrkSys", massTrkSys_log, scale_flat),
    ("TauJets/EMPOverTrkSysP", EMPOverTrkSysP_clip_log, scale_flat),
    ("TauJets/ptRatioEflowApprox", ptRatioEflowApprox_log, scale_flat),
    ("TauJets/mEflowApprox", mEflowApprox_log, scale_flat),
    ("TauJets/ptIntermediateAxis", ptIntermediateAxis_log, scale_flat)
]

charged_pfo_vars = [
    ("TauJets/charged_Phi", charged_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/charged_Eta", charged_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/charged_Pt_jet_log", charged_Pt_jet_log, scale),
    ("TauPFOs/charged_dPhi", charged_dPhi, scale),
    ("TauPFOs/charged_dEta", charged_dEta, scale),
    ("TauPFOs/charged_Pt_log", charged_Pt_log, scale)
]

neutral_pfo_vars = [
    ("TauJets/neutral_Phi", neutral_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/neutral_Eta", neutral_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/neutral_Pt_jet_log", neutral_Pt_jet_log, scale),
    ("TauPFOs/neutral_dPhi", neutral_dPhi, scale),
    ("TauPFOs/neutral_dEta", neutral_dEta, scale),
    ("TauPFOs/neutral_Pt_log", neutral_Pt_log, scale),
    ("TauPFOs/neutralPi0BDT", None, None),
    ("TauPFOs/neutralNHitsInEM1", None, None)
]

neutral_pfo_w_moment_vars = [
    ("TauJets/neutral_Phi", neutral_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/neutral_Eta", neutral_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/neutral_Pt_jet_log", neutral_Pt_jet_log, scale),
    ("TauPFOs/neutral_dPhi", neutral_dPhi, scale),
    ("TauPFOs/neutral_dEta", neutral_dEta, scale),
    ("TauPFOs/neutral_Pt_log", neutral_Pt_log, scale),
    ("TauPFOs/neutralPi0BDT", None, None),
    ("TauPFOs/neutralNHitsInEM1", None, None),
    ("TauPFOs/neutral_SECOND_R", neutral_SECOND_R_log, scale),
    ("TauPFOs/neutral_secondEtaWRTClusterPosition_EM1",
     neutral_secondEtaWRTClusterPosition_EM1_log, scale),
    ("TauPFOs/neutral_NPosECells_EM1", None, scale),
    ("TauPFOs/neutral_ENG_FRAC_CORE", None, None),
    ("TauPFOs/neutral_energyfrac_EM2", None, None)
]

neutral_pfo_bdtsort_vars = [
    ("TauJets/neutral_Phi_BDTSort", neutral_Phi_bdtsort,
     partial(constant_scale, scale=np.pi)),
    ("TauJets/neutral_Eta_BDTSort", neutral_Eta_bdtsort,
     partial(constant_scale, scale=2.5)),
    ("TauJets/neutral_Pt_jet_log_BDTSort", neutral_Pt_jet_log_bdtsort, scale),
    ("TauPFOs/neutral_dPhi_BDTSort", neutral_dPhi_bdtsort, scale),
    ("TauPFOs/neutral_dEta_BDTSort", neutral_dEta_bdtsort, scale),
    ("TauPFOs/neutral_Pt_log_BDTSort", neutral_Pt_log_bdtsort, scale),
    ("TauPFOs/neutralPi0BDT_BDTSort", None, None),
    ("TauPFOs/neutralNHitsInEM1_BDTSort", None, None)
]

conversion_vars = [
    ("TauJets/conversion_Phi", conversion_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/conversion_Eta", conversion_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/conversion_Pt_jet_log", conversion_Pt_jet_log, scale),
    ("TauConv/conversion_dPhi", conversion_dPhi, scale),
    ("TauConv/conversion_dEta", conversion_dEta, scale),
    ("TauConv/conversion_Pt_log", conversion_Pt_log, scale)
]

conversion_extrapol_vars = [
    ("TauJets/conversion_Phi", conversion_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/conversion_Eta", conversion_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/conversion_Pt_jet_log", conversion_Pt_jet_log, scale),
    ("TauConv/conversion_dPhi_extrap", dPhi_extrap, scale),
    ("TauConv/conversion_dEta_extrap", dEta_extrap, scale),
    ("TauConv/conversion_Pt_log", conversion_Pt_log, scale)
]

shot_vars = [
    ("TauJets/shot_Phi", shot_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/shot_Eta", shot_Eta, partial(constant_scale, scale=2.5)),
    ("TauJets/shot_Pt_jet_log", shot_Pt_jet_log, scale),
    ("TauPFOs/shot_dPhi", shot_dPhi, scale),
    ("TauPFOs/shot_dEta", shot_dEta, scale),
    ("TauPFOs/shot_Pt_log", shot_Pt_log, scale)
]

hadr_vars = [
    ("TauJets/hadronic_Phi", hadronic_Phi, partial(constant_scale, scale=np.pi)),
    ("TauJets/hadronic_Eta", hadronic_Eta, partial(constant_scale, scale=2.50)),
    ("TauJets/hadronic_Pt_jet_log", hadronic_Pt_jet_log, scale),
    ("TauPFOs/hadronic_dPhi", hadronic_dPhi, scale),
    ("TauPFOs/hadronic_dEta", hadronic_dEta, scale),
    ("TauPFOs/hadronic_Pt_log", hadronic_Pt_log, scale)
]
