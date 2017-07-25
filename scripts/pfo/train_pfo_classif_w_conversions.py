import argparse

import numpy as np
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from rnn_tauid.models.combined  import combined_3rnn_2final_dense_multiclass

from rnn_tauid.training.load import load_data_pfo, train_test_split, preprocess, \
                                    save_preprocessing


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

    if args.conv_var:
        import imp
        var_mod = imp.load_source("conv_mod", args.conv_var)
        invars_conv = var_mod.invars
    else:
        from rnn_tauid.common.variables import conversion_vars as invars_conv


    # Variable names
    chrg_vars = [v for v, _, _ in invars_chrg]
    neut_vars = [v for v, _, _ in invars_neut]
    conv_vars = [v for v, _, _ in invars_conv]

    # Preprocessing functions
    chrg_preproc_f = [f for _, _, f in invars_chrg]
    neut_preproc_f = [f for _, _, f in invars_neut]
    conv_preproc_f = [f for _, _, f in invars_conv]

    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.data, "r", **h5file) as data:
        ldata = len(data["TauJets/pt"])
        idx = int(args.fraction * ldata)

        print("Loading data [:{}]".format(idx))

        # nTracks selection
        nTracks = data["TauJets/nTracks"][:idx]
        mask = (nTracks == 1) | (nTracks == 3)

        # Load charged pfo data
        chrg_data = load_data_pfo(data, np.s_[:idx], invars_chrg, num=args.num_chrg)
        neut_data = load_data_pfo(data, np.s_[:idx], invars_neut, num=args.num_neut)
        conv_data = load_data_pfo(data, np.s_[:idx], invars_conv, num=args.num_conv)

        # Apply pt cut on neutral pfos
        if args.neut_pt_cut:
            pt_col = neut_vars.index("TauPFOs/neutral_Pt_log")
            neut_pfo_pt = neut_data.x[:, :, pt_col]
            pt_fail = neut_pfo_pt < np.log10(1000 * args.neut_pt_cut)
            neut_data.x[pt_fail] = np.nan

            del pt_col, neut_pfo_pt, pt_fail

        # Mask the nTracks selection
        from collections import namedtuple
        Data = namedtuple("Data", ["x", "y", "w"])

        chrg_data = Data(x=chrg_data.x[mask], y=chrg_data.y[mask], w=chrg_data.w[mask])
        neut_data = Data(x=neut_data.x[mask], y=neut_data.y[mask], w=neut_data.w[mask])
        conv_data = Data(x=conv_data.x[mask], y=conv_data.y[mask], w=conv_data.w[mask])

    chrg_train, chrg_test, neut_train, neut_test, conv_train, conv_test= \
        train_test_split([chrg_data, neut_data, conv_data], test_size=args.test_size)

    chrg_preproc = preprocess(chrg_train, chrg_test, chrg_preproc_f)
    neut_preproc = preprocess(neut_train, neut_test, neut_preproc_f)
    conv_preproc = preprocess(conv_train, conv_test, conv_preproc_f)

    for variables, preprocessing in [(chrg_vars, chrg_preproc),
                                     (neut_vars, neut_preproc),
                                     (conv_vars, conv_preproc)]:
        for var, (offset, scale) in zip(variables, preprocessing):
            print(var + ":")
            print("offsets:\n" + str(offset))
            print("scales:\n" + str(scale) + "\n")

    save_preprocessing(args.preprocessing_chrg, chrg_vars, chrg_preproc)
    save_preprocessing(args.preprocessing_neut, neut_vars, neut_preproc)
    save_preprocessing(args.preprocessing_conv, conv_vars, conv_preproc)

    # Setup training
    n_classes = chrg_train.y.shape[-1]
    shape_1 = chrg_train.x.shape[1:]
    shape_2 = neut_train.x.shape[1:]
    shape_3 = conv_train.x.shape[1:]
    model = combined_3rnn_2final_dense_multiclass(
        n_classes,
        shape_1, shape_2, shape_3,
        dense_units_1=args.dense_units_1,
        lstm_units_1=args.lstm_units_1,
        dense_units_2=args.dense_units_2,
        lstm_units_2=args.lstm_units_2,
        dense_units_3=args.dense_units_3,
        lstm_units_3=args.lstm_units_3,
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
    hist = model.fit([chrg_train.x, neut_train.x, conv_train.x], chrg_train.y,
                     sample_weight=chrg_train.w,
                     validation_data=([chrg_test.x, neut_test.x, conv_test.x],
                                      chrg_test.y, chrg_test.w),
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
    parser.add_argument("--model", default="model.h5")

    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--num-chrg", default=3)
    parser.add_argument("--num-neut", default=10)
    parser.add_argument("--num-conv", default=4)
    parser.add_argument("--neut-pt-cut", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dense-units-1", type=int, default=24)
    parser.add_argument("--dense-units-2", type=int, default=24)
    parser.add_argument("--dense-units-3", type=int, default=16)
    parser.add_argument("--lstm-units-1", type=int, default=24)
    parser.add_argument("--lstm-units-2", type=int, default=24)
    parser.add_argument("--lstm-units-3", type=int, default=16)
    parser.add_argument("--final-dense-units-1", type=int, default=48)
    parser.add_argument("--final-dense-units-2", type=int, default=32)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--chrg-var", default=None)
    parser.add_argument("--neut-var", default=None)
    parser.add_argument("--conv-var", default=None)

    args = parser.parse_args()
    main(args)
