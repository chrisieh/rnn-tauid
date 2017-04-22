from functools import partial

from rnn_tauid.common.preprocessing import robust_scale, constant_scale


track_vars = [
    ("TauTracks.ptfrac", partial(constant_scale, scale=1.5)),
    ("TauTracks.qOverP", partial(robust_scale, median=False)),
    ("TauTracks.d0", partial(robust_scale, median=False)),
    ("TauTracks.z0sinThetaTJVA", partial(robust_scale, median=False)),
    ("TauTracks.rConvII", partial(robust_scale, median=False)),
    ("TauTracks.dRJetSeedAxis", partial(constant_scale, scale=0.4)),
    ("TauTracks.eProbabilityHT", None),
    ("TauTracks.nInnermostPixelHits", partial(constant_scale, scale=3)),
    ("TauTracks.nPixelHits", partial(constant_scale, scale=11)),
    ("TauTracks.nSiHits", partial(constant_scale, scale=25))
]

cluster_vars = [
    ("TauClusters.et", partial(robust_scale, median=False,
                               low_perc=0, high_perc=50)),
    ("TauClusters.psfrac", None),
    ("TauClusters.em1frac", None),
    ("TauClusters.em2frac", None),
    ("TauClusters.em3frac", None),
    ("TauClusters.dRJetSeedAxis", partial(constant_scale, scale=0.4)),
    ("TauClusters.EM_PROBABILITY", None),
    ("TauClusters.SECOND_R", partial(robust_scale, median=False,
                                     low_perc=0, high_perc=50)),
    ("TauClusters.SECOND_LAMBDA", partial(robust_scale, median=False,
                                          low_perc=0, high_perc=50)),
    ("TauClusters.FIRST_ENG_DENS", partial(robust_scale, median=False,
                                           low_perc=0, high_perc=50)),
    ("TauClusters.CENTER_LAMBDA", partial(robust_scale, median=False,
                                          low_perc=0, high_perc=50)),
    ("TauClusters.ENG_FRAC_MAX", None)
]
