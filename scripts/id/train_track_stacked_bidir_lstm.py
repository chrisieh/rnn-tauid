import argparse

import numpy as np
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from rnn_tauid.models.lstm import lstm_bidirectional_stacked
from rnn_tauid.training.load import load_data, train_test_split, preprocess, \
                                    save_preprocessing


def main(args):
    # Loads variable / preprocessing module
    if args.var_mod:
        import imp
        var_mod = imp.load_source("var_mod", args.var_mod)
        invars = var_mod.invars
    else:
        from rnn_tauid.common.variables import track_vars as invars

    # Variable names
    variables = [v for v, _, _ in invars]
    # Preprocessing functions
    f_preproc = [f for _, _, f in invars]

    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.sig, "r", **h5file) as sig, \
         h5py.File(args.bkg, "r", **h5file) as bkg:
        lsig = len(sig["TauJets/pt"])
        lbkg = len(bkg["TauJets/pt"])

        sig_idx = int(args.fraction * lsig)
        bkg_idx = int(args.fraction * lbkg)

        print("Loading sig [:{}] and bkg [:{}]".format(sig_idx, bkg_idx))
        data = load_data(sig, bkg, np.s_[:sig_idx], np.s_[:bkg_idx],
                         invars, args.num_tracks)

        # Apply pt cut
        if args.pt_cut:
            from rnn_tauid.training.load import Data
            pt_cut = float(args.pt_cut) * 1000
            pt = np.concatenate([
                sig["TauJets/pt"][:sig_idx],
                bkg["TauJets/pt"][:bkg_idx]
            ])
            assert len(pt) == len(data.x)
            pt_mask = pt < pt_cut

            data_new = Data(
                x=data.x[pt_mask],
                y=data.y[pt_mask],
                w=data.w[pt_mask]
            )

            data = data_new

    train, test = train_test_split(data, test_size=args.test_size)
    preprocessing = preprocess(train, test, f_preproc)

    for var, (offset, scale) in zip(variables, preprocessing):
        print(var + ":")
        print("offsets:\n" + str(offset))
        print("scales:\n" + str(scale) + "\n")

    save_preprocessing(args.preprocessing, variables, preprocessing)

    # Setup training
    shape = train.x.shape[1:]
    model = lstm_bidirectional_stacked(shape, dense_units=args.dense_units,
                                       lstm_units_1=args.lstm_units_1,
                                       lstm_units_2=args.lstm_units_2)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    # Configure callbacks
    callbacks = []

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, verbose=1)
    callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(
        args.model, monitor="val_loss", save_best_only=True, verbose=1)
    callbacks.append(model_checkpoint)

    if args.csv_log:
        csv_logger = CSVLogger(args.csv_log)
        callbacks.append(csv_logger)

    # Start training
    hist = model.fit(train.x, train.y, sample_weight=train.w,
                     validation_data=(test.x, test.y, test.w),
                     nb_epoch=args.epochs, batch_size=args.batch_size,
                     callbacks=callbacks, verbose=2)

    # Determine best epoch & validation loss
    val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
    print("\nMinimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig", help="Input signal")
    parser.add_argument("bkg", help="Input background")

    parser.add_argument("--preprocessing", default="preproc.h5")
    parser.add_argument("--model", default="model.h5")

    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--num-tracks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--lstm-units-1", type=int, default=24)
    parser.add_argument("--lstm-units-2", type=int, default=24)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--var-mod", default=None)
    parser.add_argument("--pt-cut", default=None)

    args = parser.parse_args()
    main(args)
