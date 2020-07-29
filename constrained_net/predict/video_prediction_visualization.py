from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path

class VideoPredictionVis:

    def __init__(self, result_dir=None, model_name=None):
        self.result_dir = result_dir
        self.model_name = model_name

    def start(self, input_file):
        df_video_pred = pd.read_csv(input_file)

        img = self.__plot_global_and_scenario_results_separated(df_video_pred)
        img.savefig(os.path.join(self.result_dir, f"{self.model_name}_v_global_and_scenario_results.png"))

        img_global = self.__plot_global(df_video_pred)
        img_global.savefig(os.path.join(self.result_dir, f"{self.model_name}_v_global_results.png"))

        img_scenario = self.__plot_scenario(df_video_pred)
        img_scenario.savefig(os.path.join(self.result_dir, f"{self.model_name}_v_scenario_results.png"))

        img_platform = self.__plot_platform(df_video_pred)
        img_platform.savefig(os.path.join(self.result_dir, f"{self.model_name}_v_platform_results.png"))

    def start_multi(self, root_dir: Path):

        """
        root_dir must consist of subdirectories where each subdir contains a "V_prediction_stats.csv" file.
        """

        FILE_NAME = "V_prediction_stats.csv"
        sub_dirs = [item for item in root_dir.glob("*") if item.is_dir()]

        model_pred_dict = {}
        for sub_dir in sub_dirs:
            file_path = sub_dir.joinpath(FILE_NAME)
            if not file_path.is_file():
                print(f"{str(file_path)} is not a file!")
                continue

            model_name = sub_dir.name
            df_file = pd.read_csv(file_path)
            model_pred_dict[model_name] = df_file

            self.model_name = model_name

            img = self.__plot_global_and_scenario_results_separated(df_file)
            img.savefig(os.path.join(root_dir, f"{self.model_name}_video_global_scenario.png"))

        img = self.__plot_multi_results_separate(model_pred_dict)
        img.savefig(os.path.join(root_dir, "video_results_separate.png"))

        img = self.__plot_video_results_combined(model_pred_dict)
        img.savefig(os.path.join(root_dir, "video_results_combined.png"))

    def __plot_global(self, df):
        plt.figure(figsize=(10, 6))

        # Number of rows in csv-file to determine length of x-axis
        xmax = len(df.index)
        num_epochs = xmax - 1  # first row is zero-row
        x_range = range(0, xmax)

        # Use max loss to determine the height of y-axis
        max_loss = df['loss'].max()
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.05 if max_loss < 1 else 0.2

        plt.plot(x_range, df['acc'], label=f"Accuracy")
        plt.plot(x_range, df['loss'], label=f"Loss")
        plt.title(f"{self.model_name} - Video Classification ({num_epochs} epochs)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.xlim(0, xmax)
        plt.ylim(0, ylim+ystep)
        plt.xticks(np.arange(0, xmax, 5))
        plt.yticks(np.arange(0, ylim+ystep, ystep))
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

        # Determine max loss in two steps (for readability)
        max_loss = max(df['loss_scenario_flat'].max(), df['loss_scenario_indoor'].max())
        max_loss = max(max_loss,  df['loss_scenario_outdoor'].max())
        # Init y-axis
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.05 if max_loss < 1 else 0.2

        # Accuracies
        plt.plot(x_range, df['acc_scenario_flat'], 'peru', label=f"Acc. flat videos")
        plt.plot(x_range, df['acc_scenario_indoor'], 'darkviolet', label=f"Acc. indoor videos")
        plt.plot(x_range, df['acc_scenario_outdoor'], 'teal', label=f"Acc. outdoor videos")

        # Loss
        plt.plot(x_range, df['loss_scenario_flat'], 'peachpuff', label=f"Loss flat videos")
        plt.plot(x_range, df['loss_scenario_indoor'], 'plum', label=f"Loss indoor videos")
        plt.plot(x_range, df['loss_scenario_outdoor'], 'paleturquoise', label=f"Loss outdoor videos")

        plt.title(f"{self.model_name} - Video Scenario Classification")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.xlim(0, xmax)
        plt.ylim(0, ylim+ystep)
        plt.xticks(np.arange(0, xmax, 5))
        plt.yticks(np.arange(0, ylim+ystep, ystep))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        return plt

    def __plot_platform(self, df):
        plt.figure(figsize=(15, 10))

        # Number of rows in csv-file to determine length of x-axis
        xmax = len(df.index)

        # Determine max loss in two steps (for readability)
        max_loss = max(df['loss_platform_WA'].max(), df['loss_platform_YT'].max())
        max_loss = max(max_loss,  df['loss_platform_original'].max())
        # Init y-axis
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.05 if max_loss < 1 else 0.2

        # Accuracies
        plt.plot(range(1, xmax+1), df['acc_platform_WA'], 'g', label=f"Acc. WA")
        plt.plot(range(1, xmax+1), df['acc_platform_YT'], 'r', label=f"Acc. YT")
        plt.plot(range(1, xmax+1), df['acc_platform_original'], 'b', label=f"Acc. Original")

        # Loss
        plt.plot(range(1, xmax+1), df['loss_platform_WA'], 'lightgreen', label=f"Loss WA")
        plt.plot(range(1, xmax+1), df['loss_platform_YT'], 'lightcoral', label=f"Loss YT")
        plt.plot(range(1, xmax+1), df['loss_platform_original'], 'lightblue', label=f"Loss Original")

        plt.title(f"{self.model_name} - Video Platform Classification")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.xlim(1, xmax+1)
        plt.ylim(0, ylim+ystep)
        plt.xticks((range(1, xmax+1)))
        plt.yticks(np.arange(0, ylim+ystep, ystep))
        plt.legend(loc="center right")

        plt.tight_layout()

        return plt

    def __plot_global_and_scenario_results_separated(self, df):
        fig = plt.figure(figsize=(10, 6))
        xmax = len(df.index)
        num_epochs = xmax - 1 # first row is zero-row
        x_range = range(0, xmax)

        ##############################################
        #                                            #
        #   Global accuracy and loss                 #
        #                                            #
        ##############################################

        # Use max loss to determine the height of y-axis
        max_loss = df['loss'].max()
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.1 if max_loss < 1 else 0.2

        plt.subplot(2, 1, 1)
        plt.plot(x_range, df['acc'], label=f"Accuracy")
        plt.plot(x_range, df['loss'], label=f"Loss")
        plt.title(f"{self.model_name} - Video Classification ({num_epochs} epochs)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.xlim(0, xmax)
        plt.ylim(0, ylim + ystep)
        plt.xticks(np.arange(0, xmax, 5))
        plt.yticks(np.arange(0, ylim + ystep, ystep))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ########################################################
        #                                                      #
        #   Scenario (flat/indoor/outdoor) accuracy and loss   #
        #                                                      #
        ########################################################

        # Determine max loss in two steps (for readability)
        max_loss = max(df['loss_scenario_flat'].max(), df['loss_scenario_indoor'].max())
        max_loss = max(max_loss, df['loss_scenario_outdoor'].max())
        # Init y-axis
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.1 if max_loss < 1 else 0.2

        plt.subplot(2, 1, 2)
        # Accuracies
        plt.plot(x_range, df['acc_scenario_flat'], 'peru', label=f"Acc. flat videos")
        plt.plot(x_range, df['acc_scenario_indoor'], 'darkviolet', label=f"Acc. indoor videos")
        plt.plot(x_range, df['acc_scenario_outdoor'], 'teal', label=f"Acc. outdoor videos")

        # Loss
        plt.plot(x_range, df['loss_scenario_flat'], 'peachpuff', label=f"Loss flat videos")
        plt.plot(x_range, df['loss_scenario_indoor'], 'plum', label=f"Loss indoor videos")
        plt.plot(x_range, df['loss_scenario_outdoor'], 'paleturquoise', label=f"Loss outdoor videos")

        plt.title(f"{self.model_name} - Video Scenario Classification ({num_epochs} epochs)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.xlim(0, xmax)
        plt.ylim(0, ylim + ystep)
        plt.xticks(np.arange(0, xmax, 5))
        plt.yticks(np.arange(0, ylim + ystep, ystep))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        return plt

    def __plot_video_results_combined(self, dict):
        fig = plt.figure(figsize=(10, 4))

        num_epochs = float("inf")
        max_loss = 0
        model_names = ""

        for key in dict.keys():
            df = dict[key]
            if len(df.index) < num_epochs:
                num_epochs = len(df.index)

            if df['loss'].max() > max_loss:
                max_loss = df['loss'].max()

            model_names = f"{model_names}|{key}"

        xmax = num_epochs + 1
        ylim = 1 if max_loss < 1 else max_loss
        ystep = 0.1 if max_loss < 1 else 0.2

        for model_name in dict.keys():
            df_pred = dict[model_name]
            plt.plot(range(1, xmax), df_pred['acc'][:num_epochs], label=f"{model_name} acc.")
            plt.plot(range(1, xmax), df_pred['loss'][:num_epochs], label=f"{model_name} loss")


        plt.title(f"{model_names} - Video Classification ({num_epochs} epochs)")
        plt.xlim(1, xmax + 1)
        plt.ylim(0, ylim + ystep)
        plt.xticks((range(1, xmax + 1)))
        plt.yticks(np.arange(0, ylim + ystep, ystep))
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        return plt

    def __plot_multi_results_separate(self, dict):
        fig = plt.figure(figsize=(10, 6))

        idx = 1
        for model_name in dict.keys():
            df_pred = dict[model_name]

            num_epochs = len(df_pred.index)
            # Number of rows in csv-file to determine length of x-axis
            xmax = num_epochs + 1

            # Use max loss to determine the height of y-axis
            max_loss = df_pred['loss'].max()
            ylim = 1 if max_loss < 1 else max_loss
            ystep = 0.1 if max_loss < 1 else 0.2

            plt.subplot(len(dict.keys()), 1, idx)
            plt.plot(range(1, xmax), df_pred['acc'], label=f"Accuracy")
            plt.plot(range(1, xmax), df_pred['loss'], label=f"Loss")
            plt.title(f"{model_name} - Video Classification ({num_epochs} epochs)")
            plt.xlim(1, xmax + 1)
            plt.ylim(0, ylim + ystep)
            plt.xticks((range(1, xmax + 1)))
            plt.yticks(np.arange(0, ylim + ystep, ystep))
            plt.grid(True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            idx += 1

        plt.tight_layout()

        return plt

    @staticmethod
    def create_cm(input_file, class_names, scenario=None, platform=None):
        # Credits to Guru

        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        df = pd.read_csv(input_file)
        # e.g. flat/indoor/outdoor
        if scenario is not None:
            df = df[df["filename"].str.contains(scenario)]

        # e.g. original/WA/YT
        if platform is not None:
            if platform == "original":
                df = df[~df["filename"].str.contains("YT") & ~df["filename"].str.contains("WA")]
            else:
                df = df[df["filename"].str.contains(platform)]

        true_labels = df['true_class']
        pred_labels = df['top1_class']

        cm_matrix = confusion_matrix(true_labels, pred_labels)

        # Creating labels for the plot
        x_ticks = [''] * len(cm_matrix)
        y_ticks = [''] * len(cm_matrix)
        for i in np.arange(0, len(cm_matrix), 1):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)

        colorbar_lbl = 'No. videos per class'
        title = f"Video Confusion Matrix"

        if platform is not None:
            title = f"{title} - {platform}"
        if scenario is not None:
            title = f"{title} - {scenario}"

        plt.figure(figsize=(6, 6))
        sn.set(font_scale=0.5)  # for label size

        df_cm = pd.DataFrame(cm_matrix, class_names, class_names)
        ax = sn.heatmap(df_cm, annot=True, square=False, cmap="YlGnBu", cbar_kws={'label': colorbar_lbl}, vmin=0, vmax=cm_matrix.max())
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.yticks(rotation=0)
        plt.title(title, pad=5)
        plt.ylabel('True Class', labelpad=5)
        plt.xlabel('Predicted Class', labelpad=5)
        plt.tight_layout()

        return plt

    @staticmethod
    def create_cm_normalized(input_file, class_names, scenario=None, platform=None):
        # Credits to Guru
        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        df = pd.read_csv(input_file)
        # e.g. flat/indoor/outdoor
        if scenario is not None:
            df = df[df["filename"].str.contains(scenario)]

        # e.g. original/WA/YT
        if platform is not None:
            if platform == "original":
                df = df[~df["filename"].str.contains("YT") & ~df["filename"].str.contains("WA")]
            else:
                df = df[df["filename"].str.contains(platform)]

        true_labels = df['true_class']
        pred_labels = df['top1_class']

        cm_matrix = confusion_matrix(true_labels, pred_labels)

        # Creating labels for the plot
        x_ticks = [''] * len(cm_matrix)
        y_ticks = [''] * len(cm_matrix)
        for i in np.arange(0, len(cm_matrix), 1):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)

        colorbar_lbl = 'Normalized num videos per class'
        title = f"Video Confusion Matrix"

        if platform is not None:
            title = f"{title} - {platform}"
        if scenario is not None:
            title = f"{title} - {scenario}"

        plt.figure(figsize=(14, 7))
        sn.set(font_scale=0.9)  # for label size

        # From the sklearn documentation (plot example)
        # Note: possible divison by zero error
        norm_cm = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
        # Round to 2 decimals
        norm_cm = np.around(norm_cm, 2)

        df_cm = pd.DataFrame(norm_cm, class_names, class_names)
        ax = sn.heatmap(df_cm, annot=True, square=False, cmap="YlGnBu", cbar_kws={'label': colorbar_lbl}, vmin=0, vmax=1)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.yticks(rotation=0)
        plt.title(title, pad=10)
        plt.ylabel('True Class', labelpad=10)
        plt.xlabel('Predicted Class', labelpad=10)
        plt.tight_layout()

        return plt

    @staticmethod
    def create_cm_normalized_thesis_ordering(input_file, class_names, scenario=None, platform=None):
        '''
            # Credits to Guru
            In this function we order camera models in an ascending way, similar as we did in the
            report/thesis. E.g. starting from iPhone 4, iPhone 4s, etc.
        '''

        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        df = pd.read_csv(input_file)
        # e.g. flat/indoor/outdoor
        if scenario is not None:
            df = df[df["filename"].str.contains(scenario.lower())]

        # e.g. original/WA/YT
        if platform is not None:
            if platform == "original":
                df = df[~df["filename"].str.contains("YT") & ~df["filename"].str.contains("WA")]
            else:
                df = df[df["filename"].str.contains(platform)]

        true_label_order = [8,
                            1,
                            9,
                            22,
                            27,
                            15,
                            12,
                            4,
                            5,
                            13,
                            16,
                            26,
                            23,
                            21,
                            2,
                            14,
                            6,
                            3,
                            18,
                            25,
                            0,
                            19,
                            10,
                            24,
                            20,
                            7,
                            11,
                            17]

        true_labels2 = df['true_class']
        pred_labels2 = df['top1_class']

        # Change labels such that the devices occur in ascending order in cm
        true_labels = [true_label_order.index(i) for i in true_labels2]
        pred_labels = [true_label_order.index(i) for i in pred_labels2]

        cm_matrix = confusion_matrix(true_labels, pred_labels)

        # Creating labels for the plot
        x_ticks = [''] * len(cm_matrix)
        y_ticks = [''] * len(cm_matrix)
        for i in np.arange(0, len(cm_matrix), 1):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)

        colorbar_lbl = 'Normalized Classification Accuracy'
        title = f"Confusion Matrix"

        if platform is not None:
            title = f"{title} - {platform}"
        if scenario is not None:
            title = f"{title} - Scenario {scenario}"

        plt.figure(figsize=(12, 7))
        sn.set(font_scale=0.7)  # for label size

        # From the sklearn documentation (plot example)
        # Note: possible divison by zero error
        norm_cm = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
        # Round to 2 decimals
        norm_cm = np.around(norm_cm, 2)

        df_cm = pd.DataFrame(norm_cm, class_names, class_names)
        ax = sn.heatmap(df_cm, annot=True, square=False, cmap="YlGnBu", cbar_kws={'label': colorbar_lbl}, vmin=0,
                        vmax=1)
        bottom, top = ax.get_ylim()
        ax.figure.axes[-1].yaxis.label.set_size(12)
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.yticks(rotation=0)
        plt.title(title, pad=10, fontsize=14)
        plt.ylabel('True Class', labelpad=10, fontsize=12)
        plt.xlabel('Predicted Class', labelpad=10, fontsize=12)
        plt.tight_layout()

        return plt

    @staticmethod
    def create_cm_small_normalized_thesis_ordering(input_file, scenario=None, platform=None):
        # Credits to Guru
        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        df = pd.read_csv(input_file)
        # e.g. flat/indoor/outdoor
        if scenario is not None:
            df = df[df["filename"].str.contains(scenario.lower())]

        # e.g. original/WA/YT
        if platform is not None:
            if platform == "original":
                df = df[~df["filename"].str.contains("YT") & ~df["filename"].str.contains("WA")]
            else:
                df = df[df["filename"].str.contains(platform)]

        true_label_order = [8,
                            1,
                            9,
                            22,
                            27,
                            15,
                            12,
                            4,
                            5,
                            13,
                            16,
                            26,
                            23,
                            21,
                            2,
                            14,
                            6,
                            3,
                            18,
                            25,
                            0,
                            19,
                            10,
                            24,
                            20,
                            7,
                            11,
                            17]

        true_labels2 = df['true_class']
        pred_labels2 = df['top1_class']

        # Change labels such that the devices occur in ascending order in cm
        true_labels = [true_label_order.index(i) for i in true_labels2]
        pred_labels = [true_label_order.index(i) for i in pred_labels2]

        print(true_labels2)
        print(true_labels)
        print(pred_labels2)
        print(pred_labels)

        cm_matrix = confusion_matrix(true_labels, pred_labels)

        # Creating labels for the plot
        x_ticks = [''] * len(cm_matrix)
        y_ticks = [''] * len(cm_matrix)
        for i in np.arange(0, len(cm_matrix), 1):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)

        colorbar_lbl = 'Normalized classification accuracy'
        title = f"Confusion Matrix"

        if platform is not None:
            title = f"{title} - {platform}"
        if scenario is not None:
            title = f"{title} - Scenario {scenario}"

        plt.figure(figsize=(4, 4))
        sn.set(font_scale=0.8)  # for label size

        # From the sklearn documentation (plot example)
        # Note: possible divison by zero error
        norm_cm = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
        # Round to 2 decimals
        norm_cm = np.around(norm_cm, 2)

        #df_cm = pd.DataFrame(norm_cm, [], [])
        ax = sn.heatmap(norm_cm, annot=False, square=True, xticklabels=False, yticklabels=False, cmap="YlGnBu", cbar_kws={'label': colorbar_lbl, "shrink": .7}, vmin=0,
                        vmax=1)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.yticks(rotation=0)
        plt.title(title, pad=10)
        plt.ylabel('True Class', labelpad=10)
        plt.xlabel('Predicted Class', labelpad=10)
        plt.tight_layout()

        return plt