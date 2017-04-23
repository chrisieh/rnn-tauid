import argparse

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from rnn_tauid.models.lstm import lstm_two_branches
from rnn_tauid.training.load import load_data, train_test_split, preprocess, \
                                    save_preprocessing


def main(args):
    # Loads variable / preprocessing module
    if args.trk_var:
        import imp
        var_mod = imp.load_source("var_mod", args.trk_var)
        trk_vars = var_mod.invars
    else:
        from rnn_tauid.common.variables import track_vars as trk_vars

    if args.cls_var:
        import imp
        var_mod = imp.load_source("var_mod", args.cls_var)
        cls_vars = var_mod.invars
    else:
        from rnn_tauid.common.variables import cluster_vars as cls_vars

    trk_variables = [var for var, func in trk_vars]
    trk_funcs = [func for var, func in trk_vars]

    cls_variables = [var for var, func in cls_vars]
    cls_funcs = [func for var, func in cls_vars]

    trk_data = load_data(args.tracks, trk_variables, num=args.num_tracks)
    cls_data = load_data(args.clusters, cls_variables, num=args.num_clusters)

    assert np.all(trk_data.y == cls_data.y)
    assert np.all(trk_data.w == cls_data.w)

    trk_train, trk_test = train_test_split(trk_data, test_size=args.test_size)
    cls_train, cls_test = train_test_split(cls_data, test_size=args.test_size)

    trk_preprocessing = preprocess(trk_train, trk_test, trk_funcs)
    cls_preprocessing = preprocess(cls_train, cls_test, cls_funcs)
    for var, preproc in [(trk_variables, trk_preprocessing),
                         (cls_variables, cls_preprocessing)]:
        for var, (offset, scale) in zip(var, preproc):
            print(var + ":")
            print("offsets:\n" + str(offset))
            print("scales:\n" + str(scale) + "\n")

    save_preprocessing(args.track_preproc, trk_variables, trk_preprocessing)
    save_preprocessing(args.cluster_preproc, cls_variables, cls_preprocessing)

    # Setup training
    shape_1 = trk_train.x.shape[1:]
    shape_2 = cls_train.x.shape[1:]
    model = lstm_two_branches(shape_1, shape_2,
                              dense_units_1=32, dense_units_2=32,
                              lstm_units_1=32, lstm_units_2=24)
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
    hist = model.fit([trk_train.x, cls_train.x], trk_train.y,
                     sample_weight=trk_train.w,
                     validation_data=([trk_test.x, cls_test.x],
                                      trk_test.y, trk_test.w),
                     nb_epoch=args.epochs, batch_size=args.batch_size,
                     callbacks=callbacks, verbose=2)

    # Determine best epoch & validation loss
    val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
    print("\nMinimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", help="Track input data")
    parser.add_argument("clusters", help="Cluster input data")
    parser.add_argument("track_preproc", help="Track preprocessing")
    parser.add_argument("cluster_preproc", help="Cluster preprocessing")
    parser.add_argument("model", help="Model file")
    parser.add_argument("--num-tracks", default=None)
    parser.add_argument("--num-clusters", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--trk-var", default=None)
    parser.add_argument("--cls-var", default=None)

    args = parser.parse_args()

    if args.num_tracks:
        args.num_tracks = int(args.num_tracks)
    if args.num_clusters:
        args.num_clusters = int(args.num_clusters)

    main(args)
