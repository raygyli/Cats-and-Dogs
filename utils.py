import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess(images, labels):
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)
    assert len(config["input_shape"]) == 3
    images = tf.image.resize(images, config["input_shape"][:2])
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

def tfds_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy())
        labels.append(label.numpy())
    images = np.array(images)
    labels = np.array(images)
    return images, labels

def visualize_data(dataset, num_images=1):
    plt.ion()
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    if num_images == 1:
        axes = [axes]
    for i, (img, label) in enumerate(dataset.take(num_images)):
        if img.shape[0] > 1:
            img = img[0]
        axes[i].imshow(img.numpy())
    
    plt.show()

def split(dataset: tf.data.Dataset):
    total_size = len(list(dataset))
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train = list(dataset.take(train_size).as_numpy_iterator())
    test_val = dataset.skip(train_size)
    valid = list(test_val.take(val_size).as_numpy_iterator())
    test = list(test_val.skip(val_size).as_numpy_iterator())

    return train, valid, test
