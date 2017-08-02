import argparse
import logging

import numpy as np
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from rnn_tauid.models.combined  import combined_4rnn_2final_dense_multiclass

from rnn_tauid.training.load import load_data_pfo, train_test_split, preprocess, \
                                    save_preprocessing

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("train")

def main(args):
    log.debug("Entering main")

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
        from rnn_tauid.common.variables import neutral_pfo_w_moment_vars as invars_neut

    if args.conv_var:
        import imp
        var_mod = imp.load_source("conv_mod", args.conv_var)
        invars_conv = var_mod.invars
    else:
        from rnn_tauid.common.variables import conversion_vars as invars_conv

    if args.shot_var:
        import imp
        var_mod = imp.load_source("shot_mod", args.shot_var)
        invars_shot = var_mod.invars
    else:
        from rnn_tauid.common.variables import shot_vars as invars_shot

    # Variable names
    chrg_vars = [v for v, _, _ in invars_chrg]
    neut_vars = [v for v, _, _ in invars_neut]
    conv_vars = [v for v, _, _ in invars_conv]
    shot_vars = [v for v, _, _ in invars_shot]

    # Preprocessing functions
    chrg_preproc_f = [f for _, _, f in invars_chrg]
    neut_preproc_f = [f for _, _, f in invars_neut]
    conv_preproc_f = [f for _, _, f in invars_conv]
    shot_preproc_f = [f for _, _, f in invars_shot]

    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        ldata = len(data["TauJets/pt"])
        idx = int(args.fraction * ldata)

        print("Loading data [:{}]".format(idx))

        # nTracks selection
        from collections import namedtuple
        Data = namedtuple("Data", ["x", "y", "w"])
        nTracks = data["TauJets/nTracks"][:idx]
        mask = (nTracks == 1) | (nTracks == 3)

        # Load charged pfo data
        log.debug("Loading charged PFOs")
        chrg_data = load_data_pfo(data, np.s_[:idx], invars_chrg, num=args.num_chrg)
        log.debug("Applying charged mask")
        chrg_data = Data(x=chrg_data.x[mask], y=chrg_data.y[mask], w=chrg_data.w[mask])

        log.debug("Loading neutral PFOs")
        neut_data = load_data_pfo(data, np.s_[:idx], invars_neut, num=args.num_neut)
        log.debug("Applying neutral mask")
        neut_data = Data(x=neut_data.x[mask], y=neut_data.y[mask], w=neut_data.w[mask])

        log.debug("Loading conversion tracks")
        conv_data = load_data_pfo(data, np.s_[:idx], invars_conv, num=args.num_conv)
        log.debug("Applying conversion mask")
        conv_data = Data(x=conv_data.x[mask], y=conv_data.y[mask], w=conv_data.w[mask])

        log.debug("Loading shots")
        shot_data = load_data_pfo(data, np.s_[:idx], invars_shot, num=args.num_shot)
        log.debug("Applying shot mask")
        shot_data = Data(x=shot_data.x[mask], y=shot_data.y[mask], w=shot_data.w[mask])

        # Apply pt cut on neutral pfos
        log.debug("Applying neutral pt cut")
        if args.neut_pt_cut:
            pt_col = neut_vars.index("TauPFOs/neutral_Pt_log")
            neut_pfo_pt = neut_data.x[..., pt_col]
            pt_fail = neut_pfo_pt < np.log10(1000 * args.neut_pt_cut)
            neut_data.x[pt_fail] = np.nan
            del pt_col, neut_pfo_pt, pt_fail

    log.debug("Creating train-test-split")
    # chrg_train, chrg_test, neut_train, neut_test, conv_train, conv_test, shot_train, shot_test = \
    #     train_test_split([chrg_data, neut_data, conv_data, shot_data], test_size=args.test_size)

    # Alternative train_test_split to avoid doubling mem use
    assert len(chrg_data.x) == len(neut_data.x)
    assert len(neut_data.x) == len(conv_data.x)
    assert len(conv_data.x) == len(shot_data.x)

    split_idx = int(args.test_size * len(chrg_data.x))
    log.debug("Split index: {}".format(split_idx))

    data_dict = { "chrg": chrg_data, "neut": neut_data, "conv": conv_data,
                  "shot": shot_data}
    split_data = {}
    for name, data in data_dict.iteritems():
        # Test split
        split_data[name + "_test"] = Data(x=data.x[:split_idx],
                                          y=data.y[:split_idx],
                                          w=data.w[:split_idx])
        # Train split
        split_data[name + "_train"] = Data(x=data.x[split_idx:],
                                           y=data.y[split_idx:],
                                           w=data.w[split_idx:])

    # Unpack split
    chrg_train, chrg_test = split_data["chrg_train"], split_data["chrg_test"]
    neut_train, neut_test = split_data["neut_train"], split_data["neut_test"]
    conv_train, conv_test = split_data["conv_train"], split_data["conv_test"]
    shot_train, shot_test = split_data["shot_train"], split_data["shot_test"]

    log.debug("Preprocess inputs")
    chrg_preproc = preprocess(chrg_train, chrg_test, chrg_preproc_f)
    neut_preproc = preprocess(neut_train, neut_test, neut_preproc_f)
    conv_preproc = preprocess(conv_train, conv_test, conv_preproc_f)
    shot_preproc = preprocess(shot_train, shot_test, shot_preproc_f)

    for variables, preprocessing in [(chrg_vars, chrg_preproc),
                                     (neut_vars, neut_preproc),
                                     (conv_vars, conv_preproc),
                                     (shot_vars, shot_preproc)]:
        for var, (offset, scale) in zip(variables, preprocessing):
            print(var + ":")
            print("offsets:\n" + str(offset))
            print("scales:\n" + str(scale) + "\n")

    log.debug("Saving preprocessing")
    save_preprocessing(args.preprocessing_chrg, chrg_vars, chrg_preproc)
    save_preprocessing(args.preprocessing_neut, neut_vars, neut_preproc)
    save_preprocessing(args.preprocessing_conv, conv_vars, conv_preproc)
    save_preprocessing(args.preprocessing_shot, shot_vars, shot_preproc)

    # Setup training
    log.debug("Initialising model")
    n_classes = chrg_train.y.shape[-1]
    shape_1 = chrg_train.x.shape[1:]
    shape_2 = neut_train.x.shape[1:]
    shape_3 = conv_train.x.shape[1:]
    shape_4 = shot_train.x.shape[1:]
    model = combined_4rnn_2final_dense_multiclass(
        n_classes,
        shape_1, shape_2, shape_3, shape_4,
        dense_units_1=args.dense_units_1,
        lstm_units_1=args.lstm_units_1,
        dense_units_2=args.dense_units_2,
        lstm_units_2=args.lstm_units_2,
        dense_units_3=args.dense_units_3,
        lstm_units_3=args.lstm_units_3,
        dense_units_4=args.dense_units_4,
        lstm_units_4=args.lstm_units_4,
        final_dense_units_1=args.final_dense_units_1,
        final_dense_units_2=args.final_dense_units_2,
        backwards=True
    )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["categorical_accuracy"])

    # Configure callbacks
    callbacks = []

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, verbose=0)
    callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(
        args.model, monitor="val_loss", save_best_only=True, verbose=0)
    callbacks.append(model_checkpoint)

    if args.csv_log:
        csv_logger = CSVLogger(args.csv_log)
        callbacks.append(csv_logger)

    # Start training
    log.debug("Start training")
    hist = model.fit([chrg_train.x, neut_train.x, conv_train.x, shot_train.x],
                     chrg_train.y, sample_weight=chrg_train.w,
                     validation_data=([chrg_test.x, neut_test.x, conv_test.x,
                                       shot_test.x], chrg_test.y, chrg_test.w),
                     nb_epoch=args.epochs, batch_size=args.batch_size,
                     callbacks=callbacks, verbose=2)

    # Determine best epoch & validation loss
    val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
    print("\nMinimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Input data")

    parser.add_argument("--preprocessing-chrg", default="preproc_chrg.h5")
    parser.add_argument("--preprocessing-neut", default="preproc_neut.h5")
    parser.add_argument("--preprocessing-conv", default="preproc_conv.h5")
    parser.add_argument("--preprocessing-shot", default="preproc_shot.h5")
    parser.add_argument("--model", default="model.h5")

    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--num-chrg", type=int, default=3)
    parser.add_argument("--num-neut", type=int, default=10)
    parser.add_argument("--num-conv", type=int, default=4)
    parser.add_argument("--num-shot", type=int, default=6)
    parser.add_argument("--neut-pt-cut", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dense-units-1", type=int, default=24)
    parser.add_argument("--dense-units-2", type=int, default=24)
    parser.add_argument("--dense-units-3", type=int, default=16)
    parser.add_argument("--dense-units-4", type=int, default=16)
    parser.add_argument("--lstm-units-1", type=int, default=24)
    parser.add_argument("--lstm-units-2", type=int, default=24)
    parser.add_argument("--lstm-units-3", type=int, default=16)
    parser.add_argument("--lstm-units-4", type=int, default=16)
    parser.add_argument("--final-dense-units-1", type=int, default=64)
    parser.add_argument("--final-dense-units-2", type=int, default=32)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--chrg-var", default=None)
    parser.add_argument("--neut-var", default=None)
    parser.add_argument("--conv-var", default=None)
    parser.add_argument("--shot-var", default=None)

    args = parser.parse_args()
    main(args)
