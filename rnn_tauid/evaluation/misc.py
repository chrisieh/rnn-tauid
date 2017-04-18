from collections import namedtuple

import numpy as np
import sklearn.metrics as metrics
from scipy.stats import binned_statistic


def roc_curve(y_true, y_score, **kwargs):
    """
    Calculates the ROC-curve (signal efficiency vs. background rejection).
    """
    fpr, tpr, thr = metrics.roc_curve(y_true, y_score, **kwargs)

    nonzero = fpr != 0
    eff = tpr[nonzero]
    rej = 1 / fpr[nonzero]

    return eff, rej


def efficiency_error(n, m):
    """
    Calculates the error of the efficiency given by 'm' passing events of a
    total of 'n' events.

    References:
    -----------
    Treatment of Errors in Efficiency Calculations (T. Ullrich, Z. Xu):
    http://th-www.if.uj.edu.pl/~erichter/dydaktyka/Dydaktyka2012/LAB-2012/0701199v1.pdf

    Error analysis for efficiency (Glen Cowan):
    https://www.pp.rhul.ac.uk/~cowan/stat/notes/efferr.pdf
    """
    return np.sqrt((m + 1) / (n + 2) * ((m + 2) / (n + 3) - (m + 1) / (n + 2)))


def binned_efficiency(x, passes, **kwargs):
    """
    Calculates the efficiency in bins of 'x'.

    Returns:
    --------
    EfficiencyResult : namedtuple [mean, std, bin_edges]
        Mean and standard deviation of the efficiency in bins of 'x' with edges
        'bin_edges'.
    """
    def eff(arr):
        m = float(np.count_nonzero(arr))
        n = float(len(arr))
        return m / n

    def deff(arr):
        m = float(np.count_nonzero(arr))
        n = float(len(arr))
        return efficiency_error(n, m)

    mean = binned_statistic(x, passes, statistic=eff, **kwargs)
    std = binned_statistic(x, passes, statistic=deff, **kwargs)

    assert np.isclose(mean.bin_edges, std.bin_edges, rtol=1e-6, atol=0)

    EfficiencyResult = namedtuple("EfficiencyResult",
                                  ["mean", "std", "bin_edges"])

    return EfficiencyResult(mean=mean.statistic, std=std.statistic,
                            bin_edges=mean.bin_edges)
