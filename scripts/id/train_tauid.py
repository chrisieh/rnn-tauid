import argparse
import sys

import numpy as np
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from rnn_tauid.models.ffnn import MLP
from rnn_tauid.training.load import load_data, train_test_split,\
                                    preprocess, save_preprocessing


def main(args):
    # Load jet variables
    if args.jet_var:
        import imp
        jet_var = imp.load_source("jet_var", args.jet_var)
        invars_ffnn = jet_var.invars
    else:
        if "1p" in args.sig.lower() and "1p" in args.bkg.lower():
            from rnn_tauid.common.variables import id1p_vars as invars_ffnn
        elif "3p" in args.sig.lower() and "3p" in args.bkg.lower():
            from rnn_tauid.common.variables import id3p_vars as invars_ffnn
        else:
            print("Could not infer prongness from sample names.")
            sys.exit()

    # Variable names
    ffnn_vars = [v for v, _, _ in invars_ffnn]

    # Preprocessing functions
    ffnn_preproc_f = [f for _, _, f in invars_ffnn]

    h5file = dict(driver="family", memb_size=10*1024**3)
    with h5py.File(args.sig, "r", **h5file) as sig, \
         h5py.File(args.bkg, "r", **h5file) as bkg:
        lsig = len(sig["TauJets/pt"])
        lbkg = len(bkg["TauJets/pt"])

        sig_idx = int(args.fraction * lsig)
        bkg_idx = int(args.fraction * lbkg)

        print("Loading sig [:{}] and bkg [:{}]".format(sig_idx, bkg_idx))

        # Load jet data
        ffnn_data = load_data(sig, bkg, np.s_[:sig_idx], np.s_[:bkg_idx],
                              invars_ffnn)

    ffnn_train, ffnn_test = \
        train_test_split(ffnn_data, test_size=args.test_size)

    ffnn_preproc = preprocess(ffnn_train, ffnn_test, ffnn_preproc_f)

    for variables, preprocessing in [(ffnn_vars, ffnn_preproc)]:
        for var, (offset, scale) in zip(variables, preprocessing):
            print(var + ":")
            print("offsets:\n" + str(offset))
            print("scales:\n" + str(scale) + "\n")

    save_preprocessing(args.preprocessing_jet, ffnn_vars, ffnn_preproc)

    # Setup training
    shape_ffnn = ffnn_train.x.shape[1:]


    model = MLP(shape_ffnn, 128, 128)

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
    hist = model.fit(
        ffnn_train.x, ffnn_train.y, sample_weight=ffnn_train.w,
        validation_data=(ffnn_test.x, ffnn_test.y, ffnn_test.w),
        nb_epoch=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=1)

    # Determine best epoch & validation loss
    val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
    print("\nMinimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sig", help="Input signal")
    parser.add_argument("bkg", help="Input background")

    parser.add_argument("--preprocessing-jet", default="preproc_jet.h5")
    parser.add_argument("--model", default="model.h5")

    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--num-tracks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--trk-var", default=None)
    parser.add_argument("--jet-var", default=None)

    args = parser.parse_args()
    main(args)
