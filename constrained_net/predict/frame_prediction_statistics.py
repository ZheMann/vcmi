import numpy as np
import os
import pandas as pd


class FramePredictionStatistics:

    def __init__(self, result_dir):
        self.result_dir = result_dir

    def start(self):
        files = self.__get_frame_pred_files()
        if len(files) == 0:
            raise ValueError(f"No csv-files found with frame(!) predictions in {self.result_dir}")
        else:
            print(f"Found {len(files)} files.")

        global_frame_statistics = []
        for file in files:
            file_statistics = self.__get_statistics(file)
            global_frame_statistics.append(file_statistics)

        df_global_statistics = pd.DataFrame(global_frame_statistics, columns=self.__get_columns())

        df_global_statistics.loc[-1] = self.__empty_row()  # adding empty row
        df_global_statistics.index = df_global_statistics.index + 1  # shifting index
        df_global_statistics = df_global_statistics.sort_index()  # sorting by index

        output_file = os.path.join(self.result_dir, f"F_prediction_stats.csv")
        df_global_statistics.to_csv(output_file, index=False)

        return output_file

    def __get_frame_pred_files(self):
        # Get all csv-files containing video predictions from input directory
        files = [f for f in os.listdir(self.result_dir) if os.path.isfile(os.path.join(self.result_dir, f))
                 and f.endswith("F_predictions.csv")]
        files = sorted(files)

        return files

    def __get_statistics(self, file):
        file_path = os.path.join(self.result_dir, file)
        df_frame_predictions = pd.read_csv(file_path)

        # Add column 'correct' with value 1 if predicted label == true class, else 0.
        df_frame_predictions["correct"] = np.where(
            df_frame_predictions["True Label"] == df_frame_predictions["Predicted Label"], 1, 0)

        # Averages total
        acc = df_frame_predictions["correct"].mean()

        filename = file
        filename_split = file.split("_")
        for split in filename_split:
            if "fm-" in split:
                filename = split

        result = [filename, acc]

        # Videos exchanged through YouTube
        df_YT = df_frame_predictions[df_frame_predictions["File"].str.contains("YT")]
        # Videos exchanged through Whatsapp
        df_WA = df_frame_predictions[df_frame_predictions["File"].str.contains("WA")]
        # Original videos
        df_original = df_frame_predictions[~df_frame_predictions["File"].str.contains("YT") &
                                           ~df_frame_predictions["File"].str.contains("WA")]

        # Accuracy per platform
        acc_WA = df_WA["correct"].mean()
        acc_YT = df_YT["correct"].mean()
        acc_original = df_original["correct"].mean()

        # Round accuracy up to 3 decimals
        acc_WA = round(acc_WA, 3)
        acc_YT = round(acc_YT, 3)
        acc_original = round(acc_original, 3)

        # fc = frame count per platform
        fc_WA = len(df_WA)
        fc_YT = len(df_YT)
        fc_original = len(df_original)

        # Videos of flat objects (e.g. walls or skies)
        df_flat = df_frame_predictions[df_frame_predictions["File"].str.contains("flat")]
        # Videos of indoor rooms for example
        df_indoor = df_frame_predictions[df_frame_predictions["File"].str.contains("indoor")]
        # Videos of outdoor objects
        df_outdoor = df_frame_predictions[df_frame_predictions["File"].str.contains("outdoor")]

        # Accuracy for different scenarios
        acc_flat = df_flat["correct"].mean()
        acc_indoor = df_indoor["correct"].mean()
        acc_outdoor = df_outdoor["correct"].mean()

        acc_flat = round(acc_flat, 3)
        acc_indoor = round(acc_indoor, 3)
        acc_outdoor = round(acc_outdoor, 3)

        fc_flat = len(df_flat)
        fc_indoor = len(df_indoor)
        fc_outdoor = len(df_outdoor)

        result.extend([acc_WA, acc_YT, acc_original, fc_WA, fc_YT, fc_original,
                       acc_flat, acc_indoor, acc_outdoor, fc_flat, fc_indoor, fc_outdoor])

        return result

    def __get_columns(self):
        return ["model", "acc", "acc_platform_WA", "acc_platform_YT", "acc_platform_original",
                "fc_platform_WA", "fc_platform_YT", "fc_platform_original",
                "acc_scenario_flat", "acc_scenario_indoor", "acc_scenario_outdoor",
                "fc_scenario_flat", "fc_scenario_indoor", "fc_scenario_outdoor"]

    def __empty_row(self):
        num_cols = len(self.__get_columns())
        empty_row = []

        for _ in range(num_cols):
            empty_row.append(0)

        return  empty_row