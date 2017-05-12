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


# Cluster variables
et_log = partial(
    log10_epsilon,xvar="TauClusters/et")

SECOND_R_log = partial(
    log10_epsilon, xvar="TauClusters/SECOND_R", epsilon=0.1)

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


track_vars = [
    ("TauTracks/pt_log", pt_log, scale),
    ("TauTracks/pt_asym", pt_asym, scale),
    ("TauTracks/d0_abs_log", d0_abs_log, scale),
    ("TauTracks/z0sinThetaTJVA_abs_log", z0sinThetaTJVA_abs_log, scale),
    ("TauTracks/dRJetSeedAxis", None, partial(constant_scale, scale=0.4)),
    ("TauTracks/eProbabilityHT", None, None),
    ("TauTracks/nInnermostPixelHits", None, partial(constant_scale, scale=3)),
    ("TauTracks/nPixelHits", None, partial(constant_scale, scale=11)),
    ("TauTracks/nSiHits", None, partial(constant_scale, scale=25))
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
