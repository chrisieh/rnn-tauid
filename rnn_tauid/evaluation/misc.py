import numpy as np
import sklearn.metrics as metrics


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
