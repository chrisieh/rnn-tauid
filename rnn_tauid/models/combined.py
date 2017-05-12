from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, \
                         TimeDistributed, merge


def combined_rnn_ffnn(
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2_1=128, dense_units_2_2=128, dense_units_2_3=16,
        backwards_1=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches
    TODO: Have auxilliary output

    Parameters:
    -----------
    input_shape_1 / input_shape_2 : tuple
        Shape of the inputs to both branches.

    dense_units_1 / dense_units_2 : integer
        Size of the dense layers.

    lstm_units_1 / lstm_units_2 : integer
        Size of the hidden state of the LSTMs.
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards_1)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    dense_2_1 = Dense(dense_units_2_1, activation="relu")(x_2)
    dense_2_2 = Dense(dense_units_2_2, activation="relu")(dense_2_1)
    dense_2_3 = Dense(dense_units_2_3, activation="relu")(dense_2_2)

    # Merge
    merged_branches = merge([lstm_1, dense_2_3], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2], output=y)