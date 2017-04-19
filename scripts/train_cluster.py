from functools import partial

import numpy as np
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint
from rnn_tauid.models.lstm import lstm_shared_weights
from rnn_tauid.common.preprocessing import robust_scale, constant_scale


n_clusters = 6
invars = [
    ("TauClusters.et", partial(robust_scale, median=False,
                               low_perc=0, high_perc=50)),
    ("TauClusters.psfrac", None),
    ("TauClusters.em1frac", None),
    ("TauClusters.em2frac", None),
    ("TauClusters.em3frac", None),
    ("TauClusters.dRJetSeedAxis", partial(constant_scale, scale=0.4)),
    ("TauClusters.EM_PROBABILITY", None),
    ("TauClusters.SECOND_R", partial(robust_scale, median=False,
                                     low_perc=0, high_perc=50)),
    ("TauClusters.SECOND_LAMBDA", partial(robust_scale, median=False,
                                          low_perc=0, high_perc=50)),
    ("TauClusters.FIRST_ENG_DENS", partial(robust_scale, median=False,
                                           low_perc=0, high_perc=50)),
    ("TauClusters.CENTER_LAMBDA", partial(robust_scale, median=False,
                                          low_perc=0, high_perc=50)),
    ("TauClusters.ENG_FRAC_MAX", None)
]

with h5py.File("cluster_features.h5", "r") as f:
    dataf = f["data"]
    label = f["label"][...]
    weight = f["weight"][...]

    shape = (len(label), n_clusters, len(invars))
    data = np.empty(shape, dtype=np.float32)

    # Load data into memory
    for i, (var, func) in enumerate(invars):
        x = dataf[var][...]
        data[..., i] = x

# Shuffle dataset, labels and weights
for arr in (data, label, weight):
    random_state = np.random.RandomState(seed=1234567890)
    random_state.shuffle(arr)

# Train/test-split
test_size = 0.25
train_size = 1.0 - test_size
a, b = int(train_size * len(data)), len(data)
train_slice = slice(0, a)
test_slice = slice(a, b)

train = data[train_slice]
train_label = label[train_slice]
train_weight = weight[train_slice]

test = data[test_slice]
test_label = label[test_slice]
test_weight = weight[test_slice]

# Preprocessing
preprocessing = {}
for i, (var, func) in enumerate(invars):
    x_train = train[..., i]
    x_test = test[..., i]

    # Scale & offset from train, apply to test
    if func:
        offset, scale = func(x_train)
        x_train -= offset
        x_train /= scale

        x_test -= offset
        x_test /= scale

        preprocessing[var] = (offset, scale)
    else:
        offset = np.zeros((n_clusters,), dtype=np.float32)
        scale = np.ones((n_clusters,), dtype=np.float32)

        preprocessing[var] = (offset, scale)

    # Replace NaNs with zero
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0

# Save offsets and scales from preprocessing
with h5py.File("cluster_preproc.h5", "w") as f:
    # Save variable names
    f["variables"] = np.array([var for var, _ in invars], "S")
    # Save preprocessing
    for var, (offset, scale) in preprocessing.items():
        f[var + "/offset"] = offset
        f[var + "/scale"] = scale

        print(var + ":")
        print("offsets:\n" + str(offset))
        print("scales:\n" + str(scale))
        print()

# Setup training
shape = train.shape[1:]
model = lstm_shared_weights(shape, dense_units=32, lstm_units=32, backwards=True)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Configure callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("cluster_model.h5",
                                   monitor="val_loss", save_best_only=True,
                                   verbose=1)

# Start training
hist = model.fit(train, train_label, sample_weight=train_weight,
                 validation_data=(test, test_label, test_weight),
                 nb_epoch=100, batch_size=256,
                 callbacks=[model_checkpoint, early_stopping], verbose=2)

# Determine best epoch & validation loss
val_loss, epoch = min(zip(hist.history["val_loss"], hist.epoch))
print("Minimum val_loss {:.5} at epoch {}".format(val_loss, epoch + 1))
