import os
import random
import shutil

"""
    This script creates a new dataset for the devices specified in DEVICES by including only valid videos.
    Valid videos are videos that have WhatsApp and YouTube versions available in addition to the native video.
    If one of the social versions is unavailable, we do not include the video. 
    
    After this script is executed, create_train_test_set.py can be used to create the train and test set.
    
    
"""

DEVICES = ['D01_Samsung_GalaxyS3Mini',
'D02_Apple_iPhone4s',
'D03_Huawei_P9',
'D04_LG_D290',
'D05_Apple_iPhone5c',
'D06_Apple_iPhone6',
'D07_Lenovo_P70A',
'D08_Samsung_GalaxyTab3',
'D09_Apple_iPhone4',
'D10_Apple_iPhone4s',
'D11_Samsung_GalaxyS3',
'D12_Sony_XperiaZ1Compact',
'D14_Apple_iPhone5c',
'D15_Apple_iPhone6',
'D16_Huawei_P9Lite',
'D18_Apple_iPhone5c',
'D19_Apple_iPhone6Plus',
'D24_Xiaomi_RedmiNote3',
'D25_OnePlus_A3000',
'D26_Samsung_GalaxyS3Mini'
'D27_Samsung_GalaxyS5',
'D28_Huawei_P8',
'D29_Apple_iPhone5',
'D30_Huawei_Honor5c',
'D31_Samsung_GalaxyS4Mini',
'D32_OnePlus_A3003',
'D33_Huawei_Ascend',
'D34_Apple_iPhone5']



ORIGINAL_CATEGORIES = ['flat', 'indoor', 'outdoor']
VIDEO_COMPRESSION_TYPES = ['original', 'WA', 'YT']
CATEGORIES = ['flat', 'indoor', 'outdoor',
              'flatYT', 'indoorYT', 'outdoorYT',
              'flatWA', 'indoorWA', 'outdoorWA']

TRAIN_FRAMES = 200
TEST_FRAMES = 200
VISION_DATASET_DIR = "G:\\Documenten\\MSc Computing Science\\Jaar 2\\Master Thesis\\VISION dataset"
VISION_FRAMES_DIR = "G:\\Documenten\\MSc Computing Science\\Jaar 2\\Master Thesis\\VISION frames"
OUTPUT_DIR = "G:\\VBDI\\datasets\\ExpIII_identical_devices"
SEED = 42


def init_data_dir():
    if not OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR is empty!")

    if not os.path.isdir(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of dataset directory")


def copy_frames(device, videos):
    device_src_path = os.path.join(VISION_FRAMES_DIR, device)
    device_dest_path = os.path.join(OUTPUT_DIR, device)

    # Create 'train' or 'test' directory
    if not os.path.isdir(device_dest_path):
        try:
            os.makedirs(device_dest_path)
        except OSError as e:
            print(e)
            raise ValueError(f"{device} | Error during creation of directory")
    else:
        print(f"{device} | Skipping device, path already exists in data set")
        return

    for video in videos:
        video_name = str(video).split(".")[0]
        video_types = get_video_compression_types(video_name)

        for video_type in video_types:
            print(f"{device} | Copying frames for video {video_type}")
            video_path = os.path.join(device_src_path, video_type)
            files = os.listdir(video_path)

            if len(files) == 0:
                print(f"No frames found for video {video_type}")
                break

            random.shuffle(files)
            for i in range(TRAIN_FRAMES):
                f_src = os.path.join(video_path, files[i])
                shutil.copy(f_src, device_dest_path)


def get_videos_by_device(device):
    #print(f"{device} | Get valid videos")
    original_valid_videos = []
    original_invalid_videos = []

    device_path = os.path.join(VISION_DATASET_DIR, device, "videos")
    video_categories = [item for item in os.listdir(device_path) if os.path.isdir(os.path.join(device_path, item))
                        and item in ORIGINAL_CATEGORIES]

    # Create list of original videos
    # We want to include at least one video per original category in train and test
    for category in video_categories:
        # Category videos
        videos = os.listdir(os.path.join(device_path, category))
        # Check if original video is exchanged via both WA and YT
        valid_videos, invalid_videos = check_valid_video(device, videos)
        # Extend list with valid videos
        original_valid_videos.extend(valid_videos)
        # Extend list with invalid videos

    return original_valid_videos, original_invalid_videos

def check_valid_video(device, original_videos, verbose=False):
    # It is considered that a video is valid if it is available for all three platforms, i.e.:
    # original, WA and YT. Otherwise, we consider it to be invalid.
    valid_videos = []
    invalid_videos = []

    device_src_path = os.path.join(VISION_FRAMES_DIR, device)
    for video in original_videos:
        video_name = str(video).split(".")[0]
        video_types = get_video_compression_types(video_name)

        valid = True
        for video_type in video_types:
            video_path = os.path.join(device_src_path, video_type)
            if not os.path.exists(video_path):
                if verbose:
                    print(f"Path {video_path} does not exists. Therefore, {video} is not valid.")
                valid = False

        if valid:
            valid_videos.append(video)
        else:
            invalid_videos.append(video)

    return valid_videos, invalid_videos


def get_video_compression_types(video_name):
    video_types = [video_name]

    for category in ORIGINAL_CATEGORIES:
        if category in video_name:
            WA = video_name.replace(category, f"{category}WA")
            YT = video_name.replace(category, f"{category}YT")
            video_types.extend([WA, YT])
            return video_types


if __name__ == "__main__":
    random.seed(SEED)

    valid_device_video_dict = {}

    import time
    t_start = time.time()
    for device in DEVICES:
        valid_videos, invalid_videos = get_videos_by_device(device)
        print(f"{device} | Start copying frames for {len(valid_videos)} videos")
        copy_frames(device, valid_videos)
        print(f"{device} | Finished ({int(time.time() - t_start)} sec.)")


