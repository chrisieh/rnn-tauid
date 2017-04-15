import numpy as np


def scale(arr, mean=True, std=True):
    offset = np.zeros(arr.shape[1], dtype=np.float32)
    scale = np.ones(arr.shape[1], dtype=np.float32)

    if mean:
        np.nanmean(arr, out=offset, axis=0)
    if std:
        np.nanstd(arr, out=scale, axis=0)

    return offset, scale


def robust_scale(arr, median=True, interquartile=True):
    offset = np.zeros(arr.shape[1], dtype=np.float32)
    scale = np.ones(arr.shape[1], dtype=np.float32)

    if median:
        np.nanmedian(arr, out=offset, axis=0)
    if interquartile:
        perc = np.nanpercentile(arr, [75.0, 25.0], axis=0)
        np.subtract.reduce(perc, out=scale)

    return offset, scale


def max_scale(arr):
    offset = np.zeros(arr.shape[1], dtype=np.float32)
    scale = np.nanmax(arr, axis=0)

    return offset, scale


def constant_scale(arr, offset=0.0, scale=1.0):
    offset = np.full(arr.shape[1], fill_value=offset, dtype=np.float32)
    scale = np.full(arr.shape[1], fill_value=scale, dtype=np.float32)

    return offset, scale
