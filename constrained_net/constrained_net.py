import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.callbacks import TensorBoard, Callback
from keras.constraints import Constraint
from keras import backend as K


class Constrained3DKernelMinimal(Constraint):
    def __call__(self, w):
        """
                This custom weight constraint implements the constrained convolutional layer for RGB-images.
                However, this is not as straightforward as it should be. Since TensorFlow prehibits us from
                assigning new values directly to the weight matrix, we need a trick to update its values.
                This trick consists of multiplying the weight matrix by so-called 'mask matrices'
                to get the desired results.

                For example, if we want to set the center values of the weight matrix to zero, we first create a
                mask matrix of the same size consisting of only ones, except at the center cells. The center cells
                will have as value one. After multiplying the weight matrix with this mask matrix,
                we obtain a 'new' weight matrix, where the center values are set to zero but with the remaining values
                untouched.

                More information about this problem:
                #https://github.com/tensorflow/tensorflow/issues/14132

                The incoming weight matrix 'w' is a 4D array of shape (x, y, z, n_filters) where (normally):
                x = 5
                y = 5
                z = 3, since we're using RGB images
                n_filters = 3

                This means there are 3 filters in total with each filter being 3-dimensional.

                This module includes the following (main) steps:
                Recall that each 3D-filter has three xy-planes (across the z-direction).
                For each (3D-)filter:
                    1. For each of the three xy-planes, set its center value to zero.
                    2. For each of the three xy-planes, normalize its values such that the sum adds up to 1
                    3. For each of the three xy-planes, set its center value to -1

               """
        w_original_shape = w.shape
        w = w * 10000  # scale by 10k to prevent numerical issues

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2 # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, y, x]
        w = tf.reshape(w, new_shape)

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum)  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 1
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        return tf.reshape(w, w_original_shape)

    def get_config(self):
        return {}


class ConstrainedNet:
    GLOBAL_SAVE_MODEL_DIR = "/data/s3523799/vbdi/models/"
    GLOBAL_TENSORBOARD_DIR = "/data/s3523799/vbdi/tensorboard/"

    def __init__(self, model_path=None, constrained_net=True):
        self.use_TensorBoard = True
        self.model = None
        self.model_path = None
        if model_path is not None:
            self.set_model(model_path)

        self.verbose = False
        self.model_name = None

        # Constrained layer properties
        self.constrained_net = constrained_net
        self.constrained_n_filters = 3
        self.constrained_kernel_size = 5

    def set_constrained_params(self, n_filters=None, kernel_size=None):
        print(f"Setting constrained params: number of filters={n_filters}, kernel size={kernel_size}")
        if n_filters is not None:
            self.constrained_n_filters = n_filters
        if kernel_size is not None:
            self.constrained_kernel_size = kernel_size

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, value):
        self.model_name = value

    def get_model(self):
        return self.model

    def set_model(self, model_path):
        # Path is e.g. ~/constrained_net/fm-e00001.h5
        path_splits = model_path.split(os.sep)
        model_name = path_splits[-2]

        self.model_path = model_path
        self.model_name = model_name

        if self.constrained_net:
            self.model = tf.keras.models.load_model(model_path, custom_objects={
                'Constrained3DKernelMinimal': Constrained3DKernelMinimal})
        else:
            self.model = tf.keras.models.load_model(model_path)

        if self.model is None:
            raise ValueError(f"Model could not be loaded from location {model_path}")

    def set_useTensorBoard(self, value): self.use_TensorBoard = value
    def set_verbose(self, value): self.verbose = value

    def __generate_model_name(self, fc_layers, fc_size, cnn_height, cnn_width):
        model_name = f"FC{fc_layers}x{fc_size}-{cnn_height}_{cnn_width}"

        if self.constrained_net:
            n_filters = self.constrained_n_filters
            kernel_s = self.constrained_kernel_size
            model_name = f"ccnn-{model_name}-f{n_filters}-k_s{kernel_s}"
        else:
            model_name = f"cnn-{model_name}"

        return model_name

    def create_model(self, num_output, fc_layers, fc_size, height=480, width=800, model_name=None):

        input_shape = (height, width, 3)
        model = Sequential()

        if self.constrained_net:
            cons_layer = Conv2D(filters=self.constrained_n_filters,
                                kernel_size=self.constrained_kernel_size,
                                strides=(1, 1),
                                input_shape=input_shape,
                                padding="valid", # Intentionally
                                kernel_constraint=Constrained3DKernelMinimal(),
                                name="constrained_layer")
            model.add(cons_layer)

        # Determine whether to use the input shape parameter
        if self.constrained_net:
            model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="same"))
        else:
            model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape))

        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        for i in range(fc_layers):
            model.add(Dense(fc_size, activation=tf.keras.activations.tanh))

        model.add(Dense(num_output, activation=tf.keras.activations.softmax))

        # See also learning rate scheduler at the bottom of this script.
        opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, decay=0.0005)

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=["acc"])

        if model_name is None:
            model_name = self.__generate_model_name(fc_layers, fc_size, height, width)

        self.model_name = model_name
        self.model = model

        return model

    def compile(self):
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95, decay=0.0005),
                      metrics=["acc"])


    def get_tensorboard_path(self):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save TensorBoard log-files.")

        # Create directory if not exists
        path = os.path.join(self.GLOBAL_TENSORBOARD_DIR, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def get_save_model_path(self, file_name):
        if self.model_name is None:
            raise ValueError("Model has no name specified. This is required in order to save checkpoints.")

        # Create directory if not exists
        path = os.path.join(self.GLOBAL_SAVE_MODEL_DIR, self.model_name)
        if not os.path.exists(path):
            os.makedirs(path)

        # Append file name and return
        return os.path.join(path, file_name)

    def __get_initial_epoch(self):
        # Means we train from scratch
        if self.model_path is None:
            return 0

        path = self.model_path
        file_name = path.split(os.sep)[-1]

        if file_name is None:
            return 0

        file_name = file_name.split(".")[0]
        splits = file_name.split("-")
        for split in splits:
            if split.startswith("e"):
                epoch = split.strip("e")
                return int(epoch)

        return 0

    def train(self, train_ds, val_ds, epochs=1):
        if self.model is None:
            raise ValueError("Cannot start training! self.model is None!")

        initial_epoch = self.__get_initial_epoch()
        epochs += initial_epoch

        callbacks = self.get_callbacks()

        self.model.fit(train_ds,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       validation_data=val_ds,
                       callbacks=callbacks,
                       workers=12,
                       use_multiprocessing=True)


    def get_callbacks(self):
        default_file_name = "fm-e{epoch:05d}.h5"
        save_model_path = self.get_save_model_path(default_file_name)

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                         verbose=1,
                                                         save_weights_only=False,
                                                         period=1)

        tensorboard_cb = TensorBoard(log_dir=self.get_tensorboard_path())
        print_lr_cb = PrintLearningRate()

        return [save_model_cb, tensorboard_cb, print_lr_cb]

    def evaluate(self, test_ds, model_path=None):
        if model_path is not None:
            self.model = tf.keras.models.load_model(model_path)
        elif self.model is None:
            raise ValueError("No model available")

        test_loss, test_acc = self.model.evaluate(test_ds)

        return (test_acc, test_loss)

    def predict(self, dataset, load_model=None):
        if load_model is not None:
            self.model = tf.keras.models.load_model(load_model)
        elif self.model is None:
            raise ValueError("No model available")

        test_ds = dataset.get_test_data()
        predicted_labels = self.model.predict_class(test_ds)
        true_labels = dataset.get_labels(test_ds)

        return (true_labels, predicted_labels)

    def print_model_summary(self):
        if self.model is None:
            print("Can't print model summary, self.model is None!")
        else:
            print(f"\nSummary of model:\n{self.model.summary()}")


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate on_epoch_end epoch {epoch}: {K.eval(lr_with_decay)}")