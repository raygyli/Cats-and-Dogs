import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from data_loader import load_data
from model import build_model
from utils import *

def train(model, X_train, y_train, num_epochs=100, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(X_train, y_train, epochs=num_epochs)
    return history

if __name__ == "__main__":
    # Set TensorFlow GPU memory allocation mode
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

    # Open configurations
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)

    # Load data
    data, info = load_data("./data")
    train_data = data["train"]
    train_data = train_data.map(preprocess).batch(config["batch_size"]).shuffle(1000)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    # Visualize
    visualize_data(train_data, config["batch_size"])

    # Split data
    X_train, X_tmp, y_train, y_tmp = train_test_split(train_data, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    del X_tmp, y_tmp

    model = build_model(tuple(config["input_shape"]))
    hist = train(model, )
