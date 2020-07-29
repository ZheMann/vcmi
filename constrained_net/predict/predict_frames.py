import os
from constrained_net.data.data_factory import DataFactory
from constrained_net.constrained_net import Constrained3DKernelMinimal
import numpy as np
import tensorflow as tf
import pandas as pd

class FramePredictor():

    def __init__(self, model_dir=None, result_dir=None, model_fname=None, constrained=False):
        self.model_dir = model_dir
        self.model_fname = model_fname
        self.result_dir = result_dir

        # Load model
        model_path = os.path.join(model_dir, model_fname)
        if constrained:
            # This is necessary to load custom objects, or in this constraints
            self.model = tf.keras.models.load_model(model_path, custom_objects={
                                                     'Constrained3DKernelMinimal': Constrained3DKernelMinimal})
        else:
            self.model = tf.keras.models.load_model(model_path)

    def start(self, test_ds, filenames):
        output_file = self.__get_output_file()
        return self.__predict_and_save(test_ds, filenames, output_file)

    def __get_output_file(self):
        output_file = f"{self.model_fname.split('.')[0]}_F_predictions.csv"
        return os.path.join(self.result_dir, output_file)

    def __predict_frames(self, test_ds):
        predictions = self.model.predict(test_ds, verbose=2)
        actual_labels = DataFactory().get_labels(test_ds)

        # Use reduction type 'None' to create array of losses for each prediction
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cce_losses = cce(actual_labels, predictions).numpy()

        # Both actual and predicted labels are in one-hot vector form
        actual_labels = [np.argmax(x) for x in actual_labels]
        predictions = [np.argmax(x) for x in predictions]
        prediction_losses = [x for x in cce_losses]

        return actual_labels, predictions, prediction_losses

    def __predict_and_save(self, test_ds, filenames, output_file):
        true_labels, predicted_labels, losses = self.__predict_frames(test_ds=test_ds)
        df = pd.DataFrame(list(zip(filenames, true_labels, predicted_labels, losses)), columns=["File", "True Label", "Predicted Label", "Loss"])
        df.to_csv(output_file, index=False)
        return output_file
