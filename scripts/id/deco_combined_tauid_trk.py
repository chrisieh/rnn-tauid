import argparse

import numpy as np
import h5py
from tqdm import tqdm
from keras.models import load_model


def main(args):
    if args.trk_var:
        import imp
        var_mod = imp.load_source("var_mod", args.variables)
    else:
        from rnn_tauid.common.variables import track_vars as invars

    # Load jet variables
    if args.jet_var:
        import imp
        jet_var = imp.load_source("jet_var", args.jet_var)
        invars_ffnn = jet_var.invars
    else:
        if "1p" in args.data.lower():
            from rnn_tauid.common.variables import id1p_vars as invars_ffnn
        elif "3p" in args.data.lower():
            from rnn_tauid.common.variables import id3p_vars as invars_ffnn
        else:
            print("Could not infer prongness from sample name.")
            sys.exit()

    # Load preprocessing rules
    with h5py.File(args.preprocessing_track, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        trk_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        trk_scale = {v: f[v + "/scale"][...] for v in pp_invars}

    with h5py.File(args.preprocessing_jet, "r") as f:
        pp_invars = np.char.decode(f["variables"][...]).tolist()
        jet_offset = {v: f[v + "/offset"][...] for v in pp_invars}
        jet_scale = {v: f[v + "/scale"][...] for v in pp_invars}


    # Load model
    model = load_model(args.model)
    num=10 # TODO: EXTRACT THIS FROM MODEL FILE

    # Load the data
    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        length = len(data["TauJets/pt"])
        n_vars = len(invars)
        n_vars_ffnn = len(invars_ffnn)

        chunksize = 500000
        chunks = [(i, min(length, i + chunksize))
                  for i in range(0, length, chunksize)]

        x = np.empty((chunksize, num, n_vars))
        x_jet = np.empty((chunksize, n_vars_ffnn))
        pred = np.empty(length, dtype=np.float32)

        for start, stop in tqdm(chunks):
            src = np.s_[start:stop, :num]
            src_jet = np.s_[start:stop]
            lslice = stop - start

            for i, (varname, func, _) in enumerate(invars):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x, source_sel=src, dest_sel=dest)
                else:
                    data[varname].read_direct(x, source_sel=src, dest_sel=dest)

                x[dest] -= trk_offset[varname]
                x[dest] /= trk_scale[varname]

            for i, (varname, func, _) in enumerate(invars_ffnn):
                dest = np.s_[:lslice, ..., i]
                if func:
                    func(data, x_jet, source_sel=src_jet, dest_sel=dest)
                else:
                    data[varname].read_direct(x_jet, source_sel=src_jet, dest_sel=dest)

                x_jet[dest] -= jet_offset[varname]
                x_jet[dest] /= jet_scale[varname]

            # Replace nans
            x[np.isnan(x)] = 0
            x_jet[np.isnan(x_jet)] = 0

            # Predict
            pred[start:stop] = model.predict([x[:lslice], x_jet[:lslice]], batch_size=256).ravel()

    with h5py.File(args.outfile, "w") as outf:
        outf["score"] = pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preprocessing_track")
    parser.add_argument("preprocessing_jet")
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--v-trk", dest="trk_var", default=None)
    parser.add_argument("--v-jet", dest="jet_var", default=None)
    parser.add_argument("-o", dest="outfile", default="pred.h5")

    args = parser.parse_args()
    main(args)
