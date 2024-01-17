import tensorflow_datasets as tfds

def load_data(data_dir):
    ds, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True, data_dir=data_dir)
    return ds, info
