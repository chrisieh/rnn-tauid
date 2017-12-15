import argparse
import json
import h5py


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="lwtnn variables template")
    parser.add_argument("outfile", help="filled variables json")
    parser.add_argument("jet_preproc")
    parser.add_argument("trk_preproc")
    parser.add_argument("cls_preproc")

    return parser.parse_args()


def get_preproc(filename, scalar=True):
    names = []
    offsets = []
    scales = []

    if scalar:
        idx = ...
    else:
        idx = 0

    with h5py.File(filename) as f:
        for v in f["variables"][...]:
            path = v.decode()
            container, name = path.split("/")
            names.append(name)
            # Converted to lwtnn convention
            offsets.append(-f[path + "/offset"][idx])
            scales.append(1.0 / f[path + "/scale"][idx])

    return names, offsets, scales


if __name__ == "__main__":
    args = get_args()

    with open(args.infile, "r") as f:
        config = json.load(f)


    jet_names, jet_offsets, jet_scales = get_preproc(args.jet_preproc)
    trk_names, trk_offsets, trk_scales = get_preproc(args.trk_preproc, scalar=False)
    cls_names, cls_offsets, cls_scales = get_preproc(args.cls_preproc, scalar=False)


    # Set output layer name
    outputs = config["outputs"]
    assert(len(outputs) == 1)

    output_layer = config["outputs"][0]

    # Layer name
    output_layer["name"] = "rnnid_output"

    # Output node names
    assert(len(output_layer["labels"]) == 1)
    output_layer["labels"][0] = "rnnid"

    # Set input layer variables

    # Set BDT input variables
    config["inputs"][0]["name"] = "id_vars"
    variables = config["inputs"][0]["variables"]
    n_vars = len(variables)

    for v, n, o, s in zip(variables, jet_names, jet_offsets, jet_scales):
        v["name"] = n
        v["offset"] = o
        v["scale"] = s


    input_sequences = config["input_sequences"]
    assert(len(input_sequences) == 2)

    # Set Track input variables
    track = input_sequences[0]
    track_vars = track["variables"]
    assert(len(track_vars) == len(trk_names))

    track["name"] = "tracks"
    for v, n, o, s in zip(track_vars, trk_names, trk_offsets, trk_scales):
        v["name"] = n
        v["offset"] = float(o)
        v["scale"] = float(s)

    # Set cluster input variables
    cluster = input_sequences[1]
    cluster_vars = cluster["variables"]
    assert(len(cluster_vars) == len(cls_names))

    cluster["name"] = "clusters"
    for v, n, o, s in zip(cluster_vars, cls_names, cls_offsets, cls_scales):
        v["name"] = n
        v["offset"] = float(o)
        v["scale"] = float(s)

    with open(args.outfile, "w") as f:
        json.dump(config, f, indent=2)
