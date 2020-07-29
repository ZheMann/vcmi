from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools

class FramePredictionVis:

    def __init__(self, result_dir=None, model_name=None):
        self.result_dir = result_dir
        self.model_name = model_name

    def start(self, input_file):
        df_frame_pred = pd.read_csv(input_file)

        img_scenario = self.__plot_scenario(df_frame_pred)
        img_scenario.savefig(os.path.join(self.result_dir, f"{self.model_name}_f_scenario_results.png"))


    def __plot_platform(self, df):
        plt.figure(figsize=(15, 10))

        # Number of rows in csv-file to determine length of x-axis
        xmax = len(df.index)

        # Platform accuracies
        plt.plot(range(1, xmax+1), df['acc_platform_WA'], 'g', label=f"Whatsapp")
        plt.plot(range(1, xmax+1), df['acc_platform_YT'], 'r', label=f"YouTube")
        plt.plot(range(1, xmax+1), df['acc_platform_original'], label=f"Original")

        plt.title(f"Frame Platform Accuracy")
        plt.xlim(1, xmax+1)
        plt.ylim(0, 1)
        plt.xticks((range(1, xmax+1)))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        return plt

    def __plot_scenario(self, df):
        plt.figure(figsize=(10, 6))

        # Number of rows in csv-file to determine length of x-axis
        xmax = len(df.index)
        num_epochs = xmax - 1  # first row is zero-row
        x_range = range(0, xmax)

        # Platform accuracies
        plt.plot(x_range, df['acc_scenario_flat'], 'g', label=f"Flat frames")
        plt.plot(x_range, df['acc_scenario_indoor'], 'r', label=f"Indoor frames")
        plt.plot(x_range, df['acc_scenario_outdoor'], label=f"Outdoor frames")

        plt.title(f"{self.model_name} - Frame Scenario Accuracy ({num_epochs} epochs)")
        plt.xlim(0, xmax)
        plt.ylim(0, 1)
        plt.xticks(x_range)
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        return plt
