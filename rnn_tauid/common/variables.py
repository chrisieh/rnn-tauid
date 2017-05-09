from functools import partial

import numpy as np
from rnn_tauid.common.preprocessing import scale, robust_scale, constant_scale


# Functons to calculate additional variables
def pt_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauTracks/pt"].read_direct(dest, source_sel=source_sel,
                                         dest_sel=dest_sel)
    np.log10(dest[dest_sel], out=dest[dest_sel])


def pt_asym(datafile, dest, source_sel=None, dest_sel=None):
    pt_track = datafile["TauTracks/pt"][source_sel]

    datafile["TauTracks/pt"].read_direct(dest, source_sel=source_sel,
                                         dest_sel=dest_sel)

    pt_jetseed = datafile["TauJets/ptJetSeed"][source_sel[0]]
    pt_jetseed = pt_jetseed[:, np.newaxis]

    dest[dest_sel] = (pt_track - pt_jetseed) / (pt_track + pt_jetseed)


def d0_abs_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauTracks/d0"].read_direct(dest, source_sel=source_sel,
                                         dest_sel=dest_sel)
    np.abs(dest[dest_sel], out=dest[dest_sel])
    np.add(dest[dest_sel], 1e-6, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def z0sinThetaTJVA_abs_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauTracks/z0sinThetaTJVA"].read_direct(dest, source_sel=source_sel,
                                                     dest_sel=dest_sel)
    np.abs(dest[dest_sel], out=dest[dest_sel])
    np.add(dest[dest_sel], 1e-6, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def et_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauClusters/et"].read_direct(dest, source_sel=source_sel,
                                           dest_sel=dest_sel)
    np.log10(dest[dest_sel], out=dest[dest_sel])


def SECOND_R_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauClusters/SECOND_R"].read_direct(dest, source_sel=source_sel,
                                                 dest_sel=dest_sel)
    np.add(dest[dest_sel], 0.1, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def SECOND_LAMBDA_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauClusters/SECOND_LAMBDA"].read_direct(dest, source_sel=source_sel,
                                                      dest_sel=dest_sel)
    np.add(dest[dest_sel], 0.1, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def FIRST_ENG_DENS_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauClusters/FIRST_ENG_DENS"].read_direct(dest, source_sel=source_sel,
                                                       dest_sel=dest_sel)
    np.add(dest[dest_sel], 1e-6, out=dest[dest_sel])
    np.log10(dest[dest_sel], out=dest[dest_sel])


def CENTER_LAMBDA_log(datafile, dest, source_sel=None, dest_sel=None):
    datafile["TauClusters/CENTER_LAMBDA"].read_direct(dest, source_sel=source_sel,
                                                      dest_sel=dest_sel)
    np.add(dest[dest_sel], 1e-6, out=dest[dest_sel])
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
