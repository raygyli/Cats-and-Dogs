import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def preprocess(images, labels):
    images = tf.image.resize(images, [32, 32])
    images = tf.cast(images, tf.float32) / 255.0
    return images

def visualize_data(dataset, num_images=1):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    if num_images == 1:
        axes = [axes]
    for i, batch in enumerate(dataset.take(num_images)):
        if batch.shape[0] > 1:
            img = batch[0]
        else:
            img = batch
        axes[i].imshow(img.numpy())
    
    plt.show()
