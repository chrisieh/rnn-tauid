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


# Track variables
pt_log = partial(
    log10_epsilon, var="TauTracks/pt")

d0_abs_log = partial(
    abs_log10_epsilon,  var="TauTracks/d0", epsilon=1e-6)

z0sinThetaTJVA_abs_log = partial(
    abs_log10_epsilon, var="TauTracks/z0sinThetaTJVA", epsilon=1e-6)


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
def Eta(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    # Hack to set nans
    datafile["TauPFOs/{}Px".format(pfo_type)].read_direct(dest, source_sel=source_sel,
                                                          dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])

    eta = datafile["TauJets/Eta"]
    np.add(dest[dest_sel], eta[source_sel[0]][:, np.newaxis], out=dest[dest_sel])


def Phi(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    # Hack to set nans
    datafile["TauPFOs/{}Px".format(pfo_type)].read_direct(dest, source_sel=source_sel,
                                                          dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])

    phi = datafile["TauJets/Phi"]
    np.add(dest[dest_sel], phi[source_sel[0]][:, np.newaxis], out=dest[dest_sel])


def dEta(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    eta_jet = datafile["TauJets/Eta"][source_sel[0]]
    datafile["TauPFOs/{}Eta".format(pfo_type)].read_direct(dest, source_sel=source_sel,
                                               dest_sel=dest_sel)
    np.subtract(dest[dest_sel], eta_jet[:, np.newaxis], out=dest[dest_sel])


def dPhi(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    phi_jet = datafile["TauJets/Phi"][source_sel[0]]
    datafile["TauPFOs/{}Phi".format(pfo_type)].read_direct(dest, source_sel=source_sel,
                                               dest_sel=dest_sel)
    np.subtract(dest[dest_sel], phi_jet[:, np.newaxis], out=dest[dest_sel])
    np.add(dest[dest_sel], np.pi, out=dest[dest_sel])
    np.fmod(dest[dest_sel], 2 * np.pi, out=dest[dest_sel])
    np.subtract(dest[dest_sel], np.pi, out=dest[dest_sel])


def Pt_log(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    px = datafile["TauPFOs/{}Px".format(pfo_type)]
    py = datafile["TauPFOs/{}Py".format(pfo_type)]
    dest[dest_sel] = np.log10(np.sqrt(px[source_sel]**2 + py[source_sel]**2))


def Pt_jet_log(datafile, dest, source_sel=None, dest_sel=None, pfo_type="charged"):
    # Hack to set nans
    datafile["TauPFOs/{}Px".format(pfo_type)].read_direct(dest, source_sel=source_sel,
                                                          dest_sel=dest_sel)
    np.multiply(dest[dest_sel], 0, out=dest[dest_sel])

    px = datafile["TauJets/Px"]
    py = datafile["TauJets/Py"]
    dest[dest_sel] = np.add(dest[dest_sel],
                            np.log10(np.sqrt(px[source_sel[0]]**2 +
                                             py[source_sel[0]]**2))[:, np.newaxis],
                            out=dest[dest_sel])


charged_Eta = partial(Eta, pfo_type="charged")
neutral_Eta = partial(Eta, pfo_type="neutral")
charged_Phi = partial(Phi, pfo_type="charged")
neutral_Phi = partial(Phi, pfo_type="neutral")
charged_dEta = partial(dEta, pfo_type="charged")
neutral_dEta = partial(dEta, pfo_type="neutral")
charged_dPhi = partial(dPhi, pfo_type="charged")
neutral_dPhi = partial(dPhi, pfo_type="neutral")
charged_Pt_log = partial(Pt_log, pfo_type="charged")
neutral_Pt_log = partial(Pt_log, pfo_type="neutral")
charged_Pt_jet_log = partial(Pt_jet_log, pfo_type="charged")
neutral_Pt_jet_log = partial(Pt_jet_log, pfo_type="neutral")


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
