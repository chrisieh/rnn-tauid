import argparse

import numpy as np
import h5py
from tqdm import tqdm
from keras.models import load_model


def main(args):
    # Loads variable / preprocessing module
    if args.chrg_var:
        import imp
        var_mod = imp.load_source("chrg_mod", args.chrg_var)
        invars_chrg = var_mod.invars
    else:
        from rnn_tauid.common.variables import charged_pfo_vars as invars_chrg

    if args.neut_var:
        import imp
        var_mod = imp.load_source("neut_mod", args.neut_var)
        invars_neut = var_mod.invars
    else:
        from rnn_tauid.common.variables import neutral_pfo_vars as invars_neut

    # Load preprocessing rules
    with h5py.File(args.preprocessing_chrg, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        chrg_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        chrg_scale = {v: f[v + "/scale"][...] for v in pp_invars}

    with h5py.File(args.preprocessing_neut, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        neut_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        neut_scale = {v: f[v + "/scale"][...] for v in pp_invars}


    # Load model
    model = load_model(args.model)
    num_chrg = 3
    num_neut = 10
    num_output = 5

    # Load the data
    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        length = len(data["TauJets/pt"])
        n_vars_chrg = len(invars_chrg)
        n_vars_neut = len(invars_neut)

        chunksize = 500000
        chunks = [(i, min(length, i + chunksize))
                  for i in range(0, length, chunksize)]

        x_chrg = np.empty((chunksize, num_chrg, n_vars_chrg))
        x_neut = np.empty((chunksize, num_neut, n_vars_neut))

        pred = np.empty((length, num_output), dtype=np.float32)

        for start, stop in tqdm(chunks):
            src_chrg = np.s_[start:stop, :num_chrg]
            src_neut = np.s_[start:stop, :num_neut]
            lslice = stop - start

            for i, (varname, func, _) in enumerate(invars_chrg):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x_chrg, source_sel=src_chrg, dest_sel=dest)
                else:
                    data[varname].read_direct(x_chrg, source_sel=src_chrg, dest_sel=dest)

                x_chrg[dest] -= chrg_offset[varname]
                x_chrg[dest] /= chrg_scale[varname]

            for i, (varname, func, _) in enumerate(invars_neut):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x_neut, source_sel=src_neut, dest_sel=dest)
                else:
                    data[varname].read_direct(x_neut, source_sel=src_neut, dest_sel=dest)

                x_neut[dest] -= neut_offset[varname]
                x_neut[dest] /= neut_scale[varname]

            # Replace nans
            x_chrg[np.isnan(x_chrg)] = 0
            x_neut[np.isnan(x_neut)] = 0

            # Predict
            pred[start:stop] = model.predict(
                [x_chrg[:lslice], x_neut[:lslice]],
                batch_size=256)

    with h5py.File(args.outfile, "w") as outf:
        outf["score"] = pred
        outf["class"] = np.argmax(pred, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessing_chrg")
    parser.add_argument("preprocessing_neut")
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--v-chrg", dest="chrg_var", default=None)
    parser.add_argument("--v-neut", dest="neut_var", default=None)
    parser.add_argument("-o", dest="outfile", default="pred.h5")

    args = parser.parse_args()
    main(args)
