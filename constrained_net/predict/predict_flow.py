import os
from constrained_net.data.data_factory import DataFactory
from constrained_net.predict.predict_frames import FramePredictor
from constrained_net.predict.predict_videos import VideoPredictor
from constrained_net.predict.frame_prediction_statistics import FramePredictionStatistics
from constrained_net.predict.video_prediction_statistics import VideoPredictionStatistics
from constrained_net.predict.frame_prediction_visualization import FramePredictionVis
from constrained_net.predict.video_prediction_visualization import VideoPredictionVis
import argparse

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_dir', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--models', type=str, required=False, help='Models within input dir (*.h5) to evaluate. Separate models by a ",". ')
parser.add_argument('--constrained', type=int, required=False, default=1, help='Constrained layer included (0=no, 1=yes)')
parser.add_argument('--dataset', type=str, required=True, help='Dataset to use to make predictions')
parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
parser.add_argument('--height', type=int, required=False, default=480, help='Height of CNN input dimension [default=480]')
parser.add_argument('--width', type=int, required=False, default=800, help='Width of CNN input dimension [default=800]')


def get_dataset_path(dataset):
    dataset_path = "/data/s3523799/vbdi/datasets"
    return os.path.join(dataset_path, dataset)

def get_devices(dataset_path):
    test_dir = os.path.join(dataset_path, "test")
    devices = [item for item in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, item))]
    return devices


def get_models_files(input_dir, models):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    if models:
        print(f"'Models' argument is set. Only the following models will be evaluated: {models.split(',')}")
        model_split = models.split(',')
        return [file for file in files_list if file in model_split]

    return files_list


def get_result_dir(input_dir):
    output_dir = os.path.join(input_dir, "predictions")

    # Create output directory is not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of output directory")

    frames_output_dir = os.path.join(output_dir, "frames")
    videos_output_dir = os.path.join(output_dir, "videos")
    plots_output_dir = os.path.join(output_dir, "plots")
    if not os.path.isdir(frames_output_dir):
        os.makedirs(frames_output_dir)
    if not os.path.isdir(videos_output_dir):
        os.makedirs(videos_output_dir)
    if not os.path.isdir(plots_output_dir):
        os.makedirs(plots_output_dir)

    return output_dir, frames_output_dir, videos_output_dir, plots_output_dir


if __name__ == "__main__":
    args = parser.parse_args()
    model_input_dir = args.input_dir
    models_to_process = args.models
    dataset_path = get_dataset_path(args.dataset)
    batch_size = args.batch_size
    height = args.height
    width = args.width
    constrained = args.constrained == 1

    model_name = model_input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(model_input_dir, models_to_process)
    _, frames_res_dir, videos_res_dir, plots_res_dir = get_result_dir(model_input_dir)

    print(f"Found {len(model_files)} files for model {model_name}")

    for model_file in model_files:
        print(f"{model_file} | Start prediction process")

        # Re-create dataset for each model to make sure the test-generator does not mess up.
        dataset = DataFactory(input_dir=dataset_path, batch_size=batch_size, height=height, width=width)
        filename_ds, test_ds = dataset.get_tf_test_data()

        # List containing only the file names of items in test set
        test_ds_filenames = list(filename_ds.as_numpy_iterator())

        print(f"{model_file} | Start predicting frames")
        # Predict Frames
        frame_predictor = FramePredictor(model_dir=model_input_dir, model_fname=model_file, result_dir=frames_res_dir,
                                         constrained=constrained)
        frame_pred_file = frame_predictor.start(test_ds=test_ds, filenames=test_ds_filenames)
        print(f"{model_file} | Predicting frames completed")

        print(f"{model_file} | Start predicting videos")
        # Predict Videos which is based on the predicted frames
        video_predictor = VideoPredictor(model_fname=model_file, result_dir=videos_res_dir)
        # Use frame prediction file as input
        video_pred_file = video_predictor.start(frame_pred_file)
        print(f"{model_file} | Predicting videos completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps =  FramePredictionStatistics(result_dir=frames_res_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")

    vps = VideoPredictionStatistics(result_dir=videos_res_dir)
    video_stats = vps.start()
    print(f"Video Prediction Statistics Completed")

    fpv = FramePredictionVis(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")

    vpv = VideoPredictionVis(result_dir=plots_res_dir, model_name=model_name)
    vpv.start(video_stats)
    print(f"Video Prediction Visualization Completed")













