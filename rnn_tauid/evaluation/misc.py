from collections import namedtuple

import numpy as np
import sklearn.metrics as metrics
from scipy.stats import binned_statistic


def bin_center(bins):
    """
    Returns the bin centers given by the bin edges 'bins'.
    
    Parameters:
    -----------
    bins : array (N,)
        Bin edges (right inclusive).

    Returns:
    --------
    centers : array (N-1,)
        Bin centers.
    """
    return (bins[1:] + bins[:-1]) / 2.0


def bin_width(bins):
    """
    Returns the bin widths given by the bin edges 'bins'.
    
    Parameters:
    -----------
    bins : array (N,)
        Bin edges (right inclusive).

    Returns:
    --------
    widths : array (N-1,)
        Bin widths.
    """
    return bins[1:] - bins[:-1]


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

        if n == 0:
            return np.nan
        
        return m / n

    def deff(arr):
        m = float(np.count_nonzero(arr))
        n = float(len(arr))

        if n == 0:
            return np.nan
        
        return efficiency_error(n, m)

    mean = binned_statistic(x, passes, statistic=eff, **kwargs)
    std = binned_statistic(x, passes, statistic=deff, **kwargs)

    EfficiencyResult = namedtuple("EfficiencyResult",
                                  ["mean", "std", "bin_edges"])

    return EfficiencyResult(mean=mean.statistic, std=std.statistic,
                            bin_edges=mean.bin_edges)


def binned_rejection(eff):
    """
    Calcualtes the rejection from a 'EfficiencyResult'.
    
    Parameters:
    -----------
    eff : EfficiencyResult
        Efficiencies to be used for calculating rejection.
    """
    mean = 1.0 / eff.mean
    std = eff.std / eff.mean ** 2
    
    RejectionResult = namedtuple("RejectionResult",
                                 ["mean", "std", "bin_edges"])

    return RejectionResult(mean=mean, std=std, bin_edges=bin_edges.copy())
