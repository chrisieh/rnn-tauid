from keras.models import Model
from keras.layers import Input, Dense


def MLP(input_shape, dense_units_1, dense_units_2):
    x = Input(shape=input_shape)
    dense_1 = Dense(dense_units_1, activation="relu")(x)
    dense_2 = Dense(dense_units_2, activation="relu")(dense_1)
    y = Dense(1, activation="sigmoid")(dense_2)

    return Model(input=x, output=y)
