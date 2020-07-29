import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import time
AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataFactory:

    def __init__(self, width=800, height=480, batch_size=32, input_dir=None, test_dir_suffix=""):
        self.data_dir = pathlib.Path(input_dir)
        self.train_dir = pathlib.Path(os.path.join(self.data_dir, "train"))
        self.test_dir = pathlib.Path(os.path.join(self.data_dir, f"test{test_dir_suffix}"))

        self.device_types = np.array([item.name for item in self.train_dir.glob('*')])
        self.train_image_count = len(list(self.train_dir.glob('*/*.jpg')))
        self.test_image_count = len(list(self.test_dir.glob('*/*.jpg')))

        self.batch_size = batch_size
        self.img_width = width
        self.img_height = height
        self.channels = 3

        # To allow reproducability
        self.seed = 42

        class_names = sorted(self.train_dir.glob("*"))
        self.class_names = np.array([x.name for x in class_names])

    def get_class_names(self):
        return self.class_names

    def get_tf_input_dim(self):
        return tf.constant((self.img_height, self.img_width), tf.dtypes.int32)

    def device_count(self):
        return len(self.device_types)

    def get_image_count(self, type="train"):
        if type == "train":
            return self.train_image_count
        else:
            return self.test_image_count

    def get_batch_size(self):
        return self.batch_size

    def print_info(self):
        print("Devices: ", self.device_types)
        print("Training images: ", self.train_image_count)
        print("Test images: ", self.test_image_count)

    def process_path(self, file_path):
        """
              Credits: https://github.com/bgswaroop/signature-camera-detection/blob/master/main/data/signature_net_data.py
              Prepares a (image, label) pair from the provided file path
              :param file_path: full path of the file
              :return: (image, label) pair
              """
        label = self.get_label(file_path)
        img = self.load_img(file_path)

        return img, label

    def get_label(self, file_path):
        """
        Credits: https://github.com/bgswaroop/signature-camera-detection/blob/master/main/data/signature_net_data.py
        This module returns an one-hot encoded vector (label) from the provided filename.
        Assuming the file path is <...>/<class_name>/<image_name.jpg>
        NOTE: This is a pure tensorflow module
        :param file_path: full file path
        :return: one hot encoded vector
        """
        file_parts = tf.strings.split(file_path, os.path.sep)
        class_name = file_parts[-2]
        one_hot_vec = tf.cast(class_name == self.class_names, dtype=tf.dtypes.float32, name="labels")

        return one_hot_vec

    def get_file_name(self, file_path):
        file_parts = tf.strings.split(file_path, os.path.sep)
        file_name = file_parts[-1]
        return file_name

    def load_img(self, file_path, resize_dim=None):
        """
        Credits: https://github.com/bgswaroop/signature-camera-detection/blob/master/main/data/signature_net_data.py
        This module reads an image into the memory using pure tensorflow functions
        The image is scaled to [0, 1] and resized to (height x width)
        :return: tensorflow image
        """
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.dtypes.float32)

        # Set to CNN Input Dimensions
        if resize_dim is None:
            resize_dim = self.get_tf_input_dim()

        img = tf.image.resize(img, size=resize_dim)

        return img

    def get_tf_train_data(self):
        t_start = time.time()
        # https://cs230.stanford.edu/blog/datapipeline/
        file_path_ds = tf.data.Dataset.list_files(str(self.train_dir / "*/*.jpg"), shuffle=True, seed=self.seed)
        print(f"Found {len(list(file_path_ds))} images in {self.train_dir} ({int(time.time() - t_start)} sec.)")

        print(f"\nPrinting first 10 elements of dataset:\n")
        for element in file_path_ds.take(10):
            print(element)

        # Load actual images and create labels accordingly
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(f"\nFinished creating labeled dataset ({int(time.time() - t_start)} sec.)\n")

        # Determine number of total elements
        num_elements = tf.data.experimental.cardinality(labeled_ds).numpy()
        print(f"\ntotal number elements: {num_elements} ({int(time.time() - t_start)} sec.)\n")

        # Set batch and prefetch preferences
        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return labeled_ds

    def get_tf_train_val_data(self, validation_split):
        t_start = time.time()
        # https://cs230.stanford.edu/blog/datapipeline/
        file_path_ds = tf.data.Dataset.list_files(str(self.train_dir / "*/*.jpg"), shuffle=True, seed=self.seed)
        print(f"Found {len(list(file_path_ds))} images in {self.train_dir} ({int(time.time() - t_start)} sec.)")

        print(f"\nPrinting first 10 elements of dataset:\n")
        for element in file_path_ds.take(10):
            print(element)

        # Load actual images and create labels accordingly
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(f"\nFinished creating labeled dataset ({int(time.time() - t_start)} sec.)\n")

        # Determine number of total elements
        num_elements = tf.data.experimental.cardinality(labeled_ds).numpy()
        print(f"\ntotal number elements: {num_elements} ({int(time.time() - t_start)} sec.)\n")

        # Create train set
        train_ds = labeled_ds.skip(count=int(validation_split * num_elements))
        # Number of train elements
        n_train_elements = tf.data.experimental.cardinality(train_ds).numpy()
        print(f"Created train ({n_train_elements}) dataset ({int(time.time() - t_start)} sec.)")

        # Create validation set
        val_ds = labeled_ds.take(count=int(validation_split * num_elements))
        # Number of validation elements
        n_val_elements = tf.data.experimental.cardinality(val_ds).numpy()
        print(f"Created val ({n_val_elements}) dataset ({int(time.time() - t_start)} sec.)")

        # Set batch and prefetch preferences
        train_ds = train_ds.batch(self.batch_size, drop_remainder=False)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Set batch and prefetch preferences
        val_ds = val_ds.batch(self.batch_size, drop_remainder=False)
        val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return train_ds, val_ds


    def get_tf_test_data(self):
        t_start = time.time()

        # Read all filenames
        tf_image_path_ds = tf.data.Dataset.list_files(str(self.test_dir / "*/*.jpg"), shuffle=False)
        print(f"Found {len(list(tf_image_path_ds))} images in {self.test_dir} ({int(time.time() - t_start)} sec.)")

        # Create labeled dataset by loading the image and estimating the label
        labeled_ds = tf_image_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print(f"Finished loading test frames ({int(time.time() - t_start)} sec.)")

        # Create dataset of file names which is necessary for evaluation
        filename_ds = tf_image_path_ds.map(self.get_file_name, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return filename_ds, labeled_ds

    @staticmethod
    def get_labels(ds):
        # Credits to Guru
        labels_ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
        ground_truth_labels = np.array(list(labels_ds.as_numpy_iterator())).astype(np.int32)
        return ground_truth_labels