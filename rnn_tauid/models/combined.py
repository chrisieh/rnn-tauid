from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, \
                         TimeDistributed, merge, Bidirectional, \
                         Dropout
from keras.constraints import max_norm


def combined_rnn_ffnn(
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2_1=128, dense_units_2_2=128, dense_units_2_3=16,
        backwards_1=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
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


def combined_rnn_ffnn_aux_loss(
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2_1=128, dense_units_2_2=128, dense_units_2_3=16,
        backwards_1=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches and auxiliary output

    Parameters:
    -----------
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

    # Auxiliary loss
    aux_out = Dense(1, activation="sigmoid")(dense_2_2)

    # Merge
    merged_branches = merge([lstm_1, dense_2_3], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2], output=[y, aux_out])


def combined_rnn_ffnn_two_output_layers(
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2_1=128, dense_units_2_2=128,
        dense_units_3_1=32,
        backwards_1=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
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

    # Merge
    merged_branches = merge([lstm_1, dense_2_2], mode="concat")
    dense_3_1 = Dense(dense_units_3_1, activation="relu")(merged_branches)
    y = Dense(1, activation="sigmoid")(dense_3_1)

    return Model(input=[x_1, x_2], output=y)


def combined_2rnn_multiclass(
        n_classes,
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2], mode="concat")

    y = Dense(n_classes, activation="softmax")(merged_branches)

    return Model(input=[x_1, x_2], output=y)


def combined_2rnn_2final_dense_multiclass(
        n_classes,
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        final_dense_units_1=32, final_dense_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2], mode="concat")

    dense_1 = Dense(final_dense_units_1, activation="tanh")(merged_branches)
    dense_2 = Dense(final_dense_units_2, activation="tanh")(dense_1)

    y = Dense(n_classes, activation="softmax")(dense_2)

    return Model(input=[x_1, x_2], output=y)


def combined_2rnn_2final_dense_multiclass_bidirectional_stacked(
        n_classes,
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2_1=32, lstm_units_2_2=32,
        final_dense_units_1=32, final_dense_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    bidirectional_lstm_2_1 = Bidirectional(
        LSTM(output_dim=lstm_units_2_1, unroll=unroll, return_sequences=True))(
            shared_dense_2)
    bidirectional_lstm_2_2 = Bidirectional(
        LSTM(output_dim=lstm_units_2_2, unroll=unroll))(
            bidirectional_lstm_2_1)

    # Merge
    merged_branches = merge([lstm_1, bidirectional_lstm_2_2], mode="concat")

    dense_1 = Dense(final_dense_units_1, activation="tanh")(merged_branches)
    dense_2 = Dense(final_dense_units_2, activation="tanh")(dense_1)

    y = Dense(n_classes, activation="softmax")(dense_2)

    return Model(input=[x_1, x_2], output=y)


def combined_3rnn_2final_dense_multiclass(
        n_classes,
        input_shape_1, input_shape_2, input_shape_3,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        dense_units_3=32, lstm_units_3=32,
        final_dense_units_1=32, final_dense_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    mask_3 = Masking(mask_value=mask_value)(x_3)
    shared_dense_3 = TimeDistributed(
        Dense(dense_units_3, activation="tanh"))(mask_3)
    lstm_3 = LSTM(output_dim=lstm_units_3, unroll=unroll,
                  go_backwards=backwards)(shared_dense_3)

    # Merge
    merged_branches = merge([lstm_1, lstm_2, lstm_3], mode="concat")

    dense_1 = Dense(final_dense_units_1, activation="tanh")(merged_branches)
    dense_2 = Dense(final_dense_units_2, activation="tanh")(dense_1)

    y = Dense(n_classes, activation="softmax")(dense_2)

    return Model(input=[x_1, x_2, x_3], output=y)


def combined_4rnn_2final_dense_multiclass(
        n_classes,
        input_shape_1, input_shape_2, input_shape_3, input_shape_4,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        dense_units_3=32, lstm_units_3=32,
        dense_units_4=32, lstm_units_4=32,
        final_dense_units_1=32, final_dense_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    mask_3 = Masking(mask_value=mask_value)(x_3)
    shared_dense_3 = TimeDistributed(
        Dense(dense_units_3, activation="tanh"))(mask_3)
    lstm_3 = LSTM(output_dim=lstm_units_3, unroll=unroll,
                  go_backwards=backwards)(shared_dense_3)

    # Branch 4
    x_4 = Input(shape=input_shape_4)
    mask_4 = Masking(mask_value=mask_value)(x_4)
    shared_dense_4 = TimeDistributed(
        Dense(dense_units_4, activation="tanh"))(mask_4)
    lstm_4 = LSTM(output_dim=lstm_units_4, unroll=unroll,
                  go_backwards=backwards)(shared_dense_4)


    # Merge
    merged_branches = merge([lstm_1, lstm_2, lstm_3, lstm_4], mode="concat")

    dense_1 = Dense(final_dense_units_1, activation="tanh")(merged_branches)
    dense_2 = Dense(final_dense_units_2, activation="tanh")(dense_1)

    y = Dense(n_classes, activation="softmax")(dense_2)

    return Model(input=[x_1, x_2, x_3, x_4], output=y)


def combined_2rnn_final_dense_multiclass(
        n_classes,
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        final_dense_units_1=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2], mode="concat")

    dense_1 = Dense(final_dense_units_1, activation="tanh")(merged_branches)

    y = Dense(n_classes, activation="softmax")(dense_1)

    return Model(input=[x_1, x_2], output=y)


def combined_2rnn_ffnn(
        input_shape_1, input_shape_2, input_shape_3,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        dense_units_3_1=128, dense_units_3_2=128, dense_units_3_3=16,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    dense_3_1 = Dense(dense_units_3_1, activation="relu")(x_3)
    dense_3_2 = Dense(dense_units_3_2, activation="relu")(dense_3_1)
    dense_3_3 = Dense(dense_units_3_3, activation="relu")(dense_3_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2, dense_3_3], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2, x_3], output=y)


def combined_2rnn_ffnn_relu(
        input_shape_1, input_shape_2, input_shape_3,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        dense_units_3_1=128, dense_units_3_2=128, dense_units_3_3=16,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="relu"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="relu"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    dense_3_1 = Dense(dense_units_3_1, activation="relu")(x_3)
    dense_3_2 = Dense(dense_units_3_2, activation="relu")(dense_3_1)
    dense_3_3 = Dense(dense_units_3_3, activation="relu")(dense_3_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2, dense_3_3], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2, x_3], output=y)


def dropout_maxnorm(
        input_shape_1, input_shape_2, input_shape_3,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        dense_units_3_1=128, dense_units_3_2=128, dense_units_3_3=16,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    d_1 = Dropout(0.2)(mask_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="relu",
              kernel_constraint=max_norm(8.0)))(d_1)
    d_2 = Dropout(0.5)(shared_dense_1)
    lstm_1 = LSTM(activation="relu", output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards, kernel_constraint=max_norm(8.0),
                  recurrent_constraint=max_norm(8.0))(d_2)
    d_3 = Dropout(0.5)(lstm_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    d_4 = Dropout(0.2)(mask_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="relu",
              kernel_constraint=max_norm(8.0)))(d_4)
    d_5 = Dropout(0.5)(shared_dense_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards, kernel_constraint=max_norm(8.0),
                  recurrent_constraint=max_norm(8.0))(d_5)
    d_6 = Dropout(0.5)(lstm_2)

    # Branch 3
    x_3 = Input(shape=input_shape_3)
    d_7 = Dropout(0.2)(x_3)
    dense_3_1 = Dense(dense_units_3_1, activation="relu",
                      kernel_constraint=max_norm(8.0))(d_7)
    d_8 = Dropout(0.5)(dense_3_1)
    dense_3_2 = Dense(dense_units_3_2, activation="relu",
                      kernel_constraint=max_norm(8.0))(d_8)
    d_9 = Dropout(0.5)(dense_3_2)
    dense_3_3 = Dense(dense_units_3_3, activation="relu",
                      kernel_constraint=max_norm(8.0))(d_9)
    d_10 = Dropout(0.5)(dense_3_3)

    # Merge
    merged_branches = merge([d_3, d_6, d_10], mode="concat")

    dense_1 = Dense(128, activation="relu", kernel_constraint=max_norm(8.0))(
        merged_branches)
    d_11 = Dropout(0.5)(dense_1)

    y = Dense(1, activation="sigmoid")(d_11)

    return Model(input=[x_1, x_2, x_3], output=y)


def combined_2rnn(
        input_shape_1, input_shape_2,
        dense_units_1=32, lstm_units_1=32,
        dense_units_2=32, lstm_units_2=32,
        backwards=False, mask_value=0.0, unroll=True):
    """
    Network with two parallel branches

    Parameters:
    -----------
    """
    # Branch 1
    x_1 = Input(shape=input_shape_1)
    mask_1 = Masking(mask_value=mask_value)(x_1)
    shared_dense_1 = TimeDistributed(
        Dense(dense_units_1, activation="tanh"))(mask_1)
    lstm_1 = LSTM(output_dim=lstm_units_1, unroll=unroll,
                  go_backwards=backwards)(shared_dense_1)

    # Branch 2
    x_2 = Input(shape=input_shape_2)
    mask_2 = Masking(mask_value=mask_value)(x_2)
    shared_dense_2 = TimeDistributed(
        Dense(dense_units_2, activation="tanh"))(mask_2)
    lstm_2 = LSTM(output_dim=lstm_units_2, unroll=unroll,
                  go_backwards=backwards)(shared_dense_2)

    # Merge
    merged_branches = merge([lstm_1, lstm_2], mode="concat")

    y = Dense(1, activation="sigmoid")(merged_branches)

    return Model(input=[x_1, x_2], output=y)
