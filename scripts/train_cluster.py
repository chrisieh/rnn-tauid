import argparse

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from rnn_tauid.models.lstm import lstm_shared_weights
from rnn_tauid.training.load import load_data, train_test_split, preprocess, \
                                    save_preprocessing


def main(args):
    # Loads variable / preprocessing module
    if args.var_mod:
        import imp
        var_mod = imp.load_source("var_mod", args.var_mod)
        invars = var_mod.invars
    else:
        from rnn_tauid.common.variables import cluster_vars as invars
        
    variables = [var for var, func in invars]
    funcs = [func for var, func in invars]

    data = load_data(args.data, variables, num=args.num_clusters)
    train, test = train_test_split(data, test_size=args.test_size)

    preprocessing = preprocess(train, test, funcs)
    for var, (offset, scale) in zip(variables, preprocessing):
        print(var + ":")
        print("offsets:\n" + str(offset))
        print("scales:\n" + str(scale) + "\n")

    save_preprocessing(args.preprocessing, variables, preprocessing)

    # Setup training
    shape = train.x.shape[1:]
    model = lstm_shared_weights(shape, dense_units=args.dense_units,
                                lstm_units=args.lstm_units)
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
    parser.add_argument("data",
                        help="Input data")
    parser.add_argument("preprocessing",
                        help="Preprocessing offsets and scales")
    parser.add_argument("model",
                        help="Model file")
    parser.add_argument("--num-clusters", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--lstm-units", type=int, default=32)
    parser.add_argument("--csv-log", default=None)
    parser.add_argument("--var-mod", default=None)
    
    args = parser.parse_args()

    if args.num_clusters:
        args.num_clusters = int(args.num_clusters)

    main(args)
