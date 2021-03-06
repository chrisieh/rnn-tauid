from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, \
                         TimeDistributed, merge, Bidirectional
from keras.layers.normalization import BatchNormalization


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
    lstm = LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards)(
        shared_dense)
    y = Dense(1, activation="sigmoid")(lstm)

    return Model(input=x, output=y)


def lstm_shared_weights_2(input_shape, dense_units=8, lstm_units=64,
                          dense_units_final=32, backwards=False,
                          mask_value=0.0, unroll=True):
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
    lstm = LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards)(
        shared_dense)
    dense = Dense(dense_units_final, activation="tanh")(lstm)
    y = Dense(1, activation="sigmoid")(dense)

    return Model(input=x, output=y)


def lstm_bidirectional(input_shape, dense_units=8, lstm_units=64,
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
    bidirectional_lstm = Bidirectional(
        LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards))(
            shared_dense)
    y = Dense(1, activation="sigmoid")(bidirectional_lstm)

    return Model(input=x, output=y)


def lstm_bidirectional_stacked(input_shape, dense_units=32, lstm_units_1=24,
                               lstm_units_2=24, mask_value=0.0, unroll=True):
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
    bidirectional_lstm_1 = Bidirectional(
        LSTM(output_dim=lstm_units_1, unroll=unroll, return_sequences=True))(
            shared_dense)
    bidirectional_lstm_2 = Bidirectional(
        LSTM(output_dim=lstm_units_2, unroll=unroll))(
            bidirectional_lstm_1)
    y = Dense(1, activation="sigmoid")(bidirectional_lstm_2)

    return Model(input=x, output=y)


def lstm_shared_ffnn(input_shape, dense_units_1=32, dense_units_2=32,
                     lstm_units=32, backwards=False, mask_value=0.0,
                     unroll=True):
    """
    Recurrent neural network with a shared feedforward neural net at input and
    a single LSTM layer

    Parameters:
    -----------
    input_shape : tuple (timesteps, variables per timestep)
        Shape of the input.
    dense_units_1 : int
        Number of units in the first dense layer with shared weights.
    dense_units_2 : int
        Number of units in the second dense layer with shared weights.
    lstm_units : int
        Number of units in the LSTM layer.

    Returns:
    --------
    model : keras-model
        The trainable model.
    """
    x = Input(shape=input_shape)
    mask = Masking(mask_value=mask_value)(x)
    shared_dense_1 = TimeDistributed(Dense(dense_units_1, activation="tanh"))(
        mask)
    shared_dense_2 = TimeDistributed(Dense(dense_units_2, activation="tanh"))(
        shared_dense_1)
    lstm = LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards)(
        shared_dense_2)
    y = Dense(1, activation="sigmoid")(lstm)

    return Model(input=x, output=y)


def lstm_two_branches(input_shape_1, input_shape_2,
                      dense_units_1=32, dense_units_2=32,
                      lstm_units_1=32, lstm_units_2=32,
                      backwards_1=False, backwards_2=False,
                      mask_value=0.0, unroll=True):
    """
    Recurrent neural network with two branches

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
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards_2)(shared_dense_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2], output=y)


def lstm_batch_norm(input_shape, dense_units=8, lstm_units=64,
                    backwards=False, mask_value=0.0, unroll=True):
    """
    Recurrent neural network with shared weights at input, batch normalization
    and a single LSTM layer

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
    bn = BatchNormalization()(shared_dense)
    lstm = LSTM(output_dim=lstm_units, unroll=unroll, go_backwards=backwards)(bn)
    y = Dense(1, activation="sigmoid")(lstm)

    return Model(input=x, output=y)


def stacked_lstm_shared_weights(input_shape, dense_units=8, lstm_units_1=64,
                                lstm_units_2=64, backwards=False,
                                mask_value=0.0, unroll=True):
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
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards, return_sequences=True)(shared_dense)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(lstm_1)
    y = Dense(1, activation="sigmoid")(lstm_2)

    return Model(input=x, output=y)
