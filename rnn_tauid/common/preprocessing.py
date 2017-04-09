import h5py
import numpy as np


def mean_std(arr):
    if len(arr.shape) == 1:
        arr = arr.reshape((-1, 1))

    offset = np.nanmean(arr, axis=0)
    scale = np.nanstd(arr, axis=0)

    return offset, scale


def median_percentile(arr):
    if len(arr.shape) == 1:
        arr = arr.reshape((-1, 1))

    offset = np.nanmedian(arr, axis=0)
    scale = np.nanstd(arr, axis=0)

    return offset, scale


def preprocess(files, start=0, stop=None):
    offset_scale = []
    for filename, groupname in files:
        with h5py.File(filename, "r") as file:
            ds = file[groupname]

            if len(ds.shape) == 1:
                shape = (1,)
            else:
                n, m = ds.shape
                shape = (1, m)

            offset = np.zeros(shape, dtype=ds.dtype)
            scale = np.ones(shape, dtype=ds.dtype)

            if len(ds.shape) == 1:
                it = [slice(start, stop)]
            else:
                it = [(slice(start, stop), i) for i in range(m)]

            for s in it:
                data = ds[s]
                view = data.view(np.float32).reshape(data.shape + (-1,))

                median = np.nanmedian(view, axis=0)
                perc_low = np.nanpercentile(view, 25.0, axis=0)
                perc_high = np.nanpercentile(view, 75.0, axis=0)

                if len(data) == 1:
                    offset_view = offset.view(np.float32)
                    scale_view = scale.view(np.float32)
                else:
                    n, m = s
                    offset_view = offset[m].view(np.float32)
                    scale_view = scale[m].view(np.float32)

                offset_view[...] = median
                # TODO: check if zero
                scale_view[...] = perc_high - perc_low

            offset_scale.append((offset, scale))

    return offset_scale
