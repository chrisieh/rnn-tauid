from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, \
                         TimeDistributed, merge


def lstm_simple(input_shape, lstm_units=128, dropout=None, backwards=False,
                mask_value=0.0, unroll=True):
    """
    Simple recurrent neural network using a single LSTM layer with dropout

    Parameters:
    -----------
    input_shape : tuple (timesteps, number of variables per timestep)
        Shape of the input given by number of tracks and number of variables per
        track.

    lstm_units : int
        Number of internal units of the LSTM.

    dropout : float (0.0, 1.0)
        Dropout fraction after the LSTM.

    Returns:
    --------
    model : keras-model
        The trainable model.
    """
    x = Input(shape=input_shape)
    mask = Masking(mask_value=mask_value)(x)
    lstm = LSTM(output_dim=lstm_units, unroll=unroll,
                go_backwards=backwards)(mask)

    if dropout:
        dropout = Dropout(dropout)(lstm)
        y = Dense(1, activation="sigmoid")(dropout)
    else:
        y = Dense(1, activation="sigmoid")(lstm)

    return Model(input=x, output=y)


def lstm_shared_weights(input_shape, dense_units=8, lstm_units=64,
                        backwards=False, mask_value=0.0, unroll=True):
    """
    Recurrent neural network with shared weights at input and a single LSTM
    layer

    Parameters:
    -----------
    input_shape : tuple (timesteps, variables per timestep)
        Shape of the input.
    dense_units : int
        Number of units of the dense layer with shared weights.
    lstm_units : int
        Number of units in the LSTM layer.

    Returns:
    --------
    model : keras-model
        The trainable model.
    """
    x = Input(shape=input_shape)
    mask = Masking(mask_value=mask_value)(x)
    shared_dense = TimeDistributed(Dense(dense_units, activation="tanh"))(mask)
    lstm = LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards)(shared_dense)
    y = Dense(1, activation="sigmoid")(lstm)

    return Model(input=x, output=y)


def lstm_two_branches(input_shape_1, input_shape_2,
                      units_1=128, units_2=128,
                      dropout_1=0.5, dropout_2=0.5,
                      interm_dense_1=16, interm_dense_2=16,
                      backwards=False, mask_value=0.0, unroll=True):
    """
    Recurrent neural network with two branches

    Parameters:
    -----------
    input_shape_1 / input_shape_2 : tuple
        Shape of the inputs to both branches.

    units_1 / units_2 : integer
        Size of the hidden state of the LSTMs.

    dropout_1 / dropout_2 : float
        Dropout after the LSTMs.

    interm_dense_1 / interm_dense_2 : int
        Size of the dense layers after the LSTM.
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    lstm_1 = LSTM(output_dim=units_1, unroll=unroll, go_backwards=backwards)(
        mask_1
    )
    dropout_1 = Dropout(dropout_1)(lstm_1)
    dense_1 = Dense(interm_dense_1, activation="relu")(dropout_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    lstm_2 = LSTM(output_dim=units_2, unroll=unroll, go_backwards=backwards)(
        mask_2
    )
    dropout_2 = Dropout(dropout_2)(lstm_2)
    dense_2 = Dense(interm_dense_2, activation="relu")(dropout_2)

    # Merge
    merge_branches = merge([dense_1, dense_2], mode="concat")

    # Final dense
    y = Dense(1, activation="sigmoid")(merge_branches)

    return Model(input=[x_1, x_2], output=y)
