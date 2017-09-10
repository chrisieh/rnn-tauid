import argparse

import numpy as np
import h5py
from tqdm import tqdm
from keras.models import load_model


def main(args):
    if args.trk_var:
        import imp
        trk_mod = imp.load_source("trk_mod", args.trk_var)
        trk_var = trk_mod.invars
    else:
        from rnn_tauid.common.variables import track_vars as trk_var

    if args.cls_var:
        import imp
        cls_mod = imp.load_source("cls_mod", args.variables)
        cls_var = cls_mod.invars
    else:
        from rnn_tauid.common.variables import cluster_vars as cls_var

    # Load preprocessing rules
    with h5py.File(args.preprocessing_track, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        trk_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        trk_scale = {v: f[v + "/scale"][...] for v in pp_invars}

    with h5py.File(args.preprocessing_cluster, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        cls_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        cls_scale = {v: f[v + "/scale"][...] for v in pp_invars}

    # Load model
    model = load_model(args.model)
    num_trk = 10
    num_cls = 6

    # Load the data
    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        length = len(data["TauJets/pt"])
        n_vars_trk = len(trk_var)
        n_vars_cls = len(cls_var)

        chunksize = 500000
        chunks = [(i, min(length, i + chunksize))
                  for i in range(0, length, chunksize)]

        x_trk = np.empty((chunksize, num_trk, n_vars_trk))
        x_cls = np.empty((chunksize, num_cls, n_vars_cls))
        pred = np.empty(length, dtype=np.float32)

        for start, stop in tqdm(chunks):
            src_trk = np.s_[start:stop, :num_trk]
            src_cls = np.s_[start:stop, :num_cls]
            lslice = stop - start

            for i, (varname, func, _) in enumerate(trk_var):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x_trk, source_sel=src_trk, dest_sel=dest)
                else:
                    data[varname].read_direct(x_trk, source_sel=src_trk, dest_sel=dest)

                x_trk[dest] -= trk_offset[varname]
                x_trk[dest] /= trk_scale[varname]

            for i, (varname, func, _) in enumerate(cls_var):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x_cls, source_sel=src_cls, dest_sel=dest)
                else:
                    data[varname].read_direct(x_cls, source_sel=src_cls, dest_sel=dest)

                x_cls[dest] -= cls_offset[varname]
                x_cls[dest] /= cls_scale[varname]

            # Replace nans
            x_trk[np.isnan(x_trk)] = 0
            x_cls[np.isnan(x_cls)] = 0

            # Predict
            pred[start:stop] = model.predict(
                [x_trk[:lslice], x_cls[:lslice]],
                batch_size=256).ravel()

    with h5py.File(args.outfile, "w") as outf:
        outf["score"] = pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessing_track")
    parser.add_argument("preprocessing_cluster")
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--v-trk", dest="trk_var", default=None)
    parser.add_argument("--v-cls", dest="cls_var", default=None)
    parser.add_argument("-o", dest="outfile", default="pred.h5")

    args = parser.parse_args()
    main(args)
