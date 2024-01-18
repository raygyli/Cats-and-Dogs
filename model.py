import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ReLU

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(inputs)
    X = ReLU()(X)
    # X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(X)
    # X = ReLU()(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=2)(X)
    X = Flatten()(X)
    flatten_shape = tf.reduce_prod(X.shape[1:])
    X = Dense(flatten_shape, activation='relu')(X)

    outputs = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=inputs, outputs=outputs)
    return model
