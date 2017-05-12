import argparse

import numpy as np
import h5py
from tqdm import tqdm
from keras.models import load_model


def main(args):
    if args.variables:
        import imp
        var_mod = imp.load_source("var_mod", args.variables)
    else:
        from rnn_tauid.common.variables import track_vars as invars

    # Variable names
    variables = [v for v, _, _ in invars]

    # Load preprocessing rules
    with h5py.File(args.preprocessing, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        offset = {v: f[v + "/offset"][...] for v in pp_invars}
        scale = {v: f[v + "/scale"][...] for v in pp_invars}

    # Load model
    model = load_model(args.model)
    num=10 # TODO: EXTRACT THIS FROM MODEL FILE

    # Load the data
    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        length = len(data["TauJets/pt"])
        n_vars = len(invars)

        chunksize = 500000
        chunks = [(i, min(length, i + chunksize))
                  for i in range(0, length, chunksize)]

        x = np.empty((chunksize, num, n_vars))
        pred = np.empty(length, dtype=np.float32)

        for start, stop in tqdm(chunks):
            src = np.s_[start:stop, :num]
            lslice = stop - start

            for i, (varname, func, _) in enumerate(invars):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x, source_sel=src, dest_sel=dest)
                else:
                    data[varname].read_direct(x, source_sel=src, dest_sel=dest)

                x[dest] -= offset[varname]
                x[dest] /= scale[varname]

            # Replace nans
            x[np.isnan(x)] = 0

            # Predict
            pred[start:stop] = model.predict(x[:lslice], batch_size=256).ravel()

    with h5py.File(args.outfile, "w") as outf:
        outf["score"] = pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessing")
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("-v", dest="variables", default=None)
    parser.add_argument("-o", dest="outfile", default="pred.h5")

    args = parser.parse_args()
    main(args)
