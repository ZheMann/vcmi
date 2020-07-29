import os
import cv2
import time
import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description='Extract and save video frames',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_dir', type=str, required=True, help='Path to VISION dataset (input directory consisting of folders per device)')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the frames')
parser.add_argument('--frames_to_save_per_video', type=int, default=1, required=False, help='Number of frames to save per video')
parser.add_argument('--save_frame_per_second', type=bool, default=False, required=False, help='Extract and save a single frame per second (overrules --frames_per_video if set to true)')
parser.add_argument('--devices', type=str, required=False, help='Only extract frames of these devices (separated by a \',\')')

def start():
    for device in DEVICES:
        print(f"\n{device} | Start")
        videos_processed = 0
        device_time_start = time.time()

        path = os.path.join(INPUT_DIR, device)

        total_nb_videos = get_total_number_of_videos(path)

        # Search for videos through all sub-directories
        for root, dirs, files in os.walk(path):
            # If no files present -> continue
            if len(files) == 0:
                continue

            for name in files:
                if name:
                    print(f"processing {name}")

                    video_time_start = time.time()

                    video_path = os.path.join(root, name)
                    # The video's name without extension is used for assigning names to the frames.
                    video_name = name.split(".")[0]
                    frames_saved = video_to_frames(video_name, video_path, device, True)
                    videos_processed += 1

                    video_time_end = time.time()
                    print(f"Finished video {video_name} ({videos_processed}/{total_nb_videos}) in {int(video_time_end - video_time_start)} seconds. {frames_saved} Frames are saved.")

        device_time_end = time.time()
        print(f"{device} | Finished {total_nb_videos} videos in {int(device_time_end - device_time_start)} seconds.")


def get_total_number_of_videos(path):
    count = 0
    for root, dirs, files in os.walk(path):
        # If no files present -> continue
        if len(files) == 0:
            continue

        for video in files:
            count+=1

    return count


def video_to_frames(video_name, video_path, device, verbose=True):
    # Create video output dir
    output_dir = os.path.join(OUTPUT_DIR, device, video_name)

    # Create directory if not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            return
    else:
        print(f"Video {video_name} already exists. Continueing.")
        return 0

    # Start capturing the feed
    cap = cv2.VideoCapture(video_path)

    # Frame rate per second
    frame_rate = np.floor(cap.get(5))

    # Total number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Calculate modulo to save frames throughout complete video, rather than frames [1:1+FRAMES_PER_VIDEO]
    mod = 1
    if video_length > FRAMES_TO_SAVE_PER_VIDEO:
        mod = video_length // FRAMES_TO_SAVE_PER_VIDEO

    number_of_frames_to_save = FRAMES_TO_SAVE_PER_VIDEO
    if SAVE_FRAME_PER_SECOND:
        number_of_frames_to_save = np.floor(video_length / frame_rate)

    if verbose:
        print(f"Video: {video_name}, #frames: {video_length}, FPS: {frame_rate}, #frames to save: {number_of_frames_to_save}.")

    frames_saved = 0
    count = 0

    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()

        # Frame is available
        if ret:
            # Get current frame id
            frame_id = cap.get(1)

            # Determine whether we have to save this frame
            if SAVE_FRAME_PER_SECOND and frame_id % frame_rate == 0:
                save_frame = True
            else:
                save_frame = frame_id % mod == 0

            # Write frame to disk
            if save_frame:
                # Check whether we have to resize or crop the frame
                cv2.imwrite(output_dir + f"/{video_name}-" + "%#05d.jpg" % frame_id, frame)
                frames_saved = frames_saved + 1
        count += 1

        if (frames_saved >= number_of_frames_to_save or count >= video_length):
            # Release the feed
            if cap.isOpened():
                cap.release()

            break

    return frames_saved

def manual():
    INPUT_DIR = "G:/Documenten/MSc Computing Science/Jaar 2/Master Thesis/VISION dataset"
    OUTPUT_DIR = "G:\\VBDI\\VISION_complete"
    DEVICES = [item for item in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, item))]
    FRAMES_TO_SAVE_PER_VIDEO = 300

    start()

if __name__ == '__main__':
    args = parser.parse_args()

    # input dir is the path to the VISION dataset in its original structure
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    FRAMES_TO_SAVE_PER_VIDEO = args.frames_to_save_per_video
    SAVE_FRAME_PER_SECOND = args.save_frame_per_second

    DEVICES = [item for item in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, item))]

    if args.devices:
        DEVICES = args.devices.split(",")

    if len(DEVICES) == 0:
        raise ValueError(f"No devices found in input directory {INPUT_DIR}.")

    print(f"Devices ({len(DEVICES)}) to process: {DEVICES}")

    start()