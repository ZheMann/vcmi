import os
import random
import shutil
from collections import namedtuple
from pathlib import Path
import time

import cv2
import numpy as np


def get_patches(img_data, std_threshold, max_num_patches):
    patches = []

    # Default patches is returned when no patches are found with a Std.Dev. lower than the threshold
    default_patch_std = np.array([float('inf'), float('inf'), float('inf')])
    default_patch = None

    patch = namedtuple('WindowSize', ['width', 'height'])(128, 128)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(128, 128)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            cropped_img = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            patch_std = np.std(cropped_img.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(patch_std, std_threshold)):
                patches.append(cropped_img)
            elif np.prod(np.less_equal(patch_std, default_patch_std)):
                default_patch_std = patch_std
                default_patch = cropped_img

    # If not a single patches has lower StdDev. than the threshold, we add the patches with the
    # lowest Std.Dev to return at least one patches.
    if len(patches) == 0:
        patches.append(default_patch)

    # Filter out excess patches
    if len(patches) > max_num_patches:
        random.seed(999)
        indices = random.sample(range(len(patches)), max_num_patches)
        patches = [patches[x] for x in indices]

    return patches


def save_patches(patches, source_img_path, destination_dir):
    for patch_id, patch in enumerate(patches, 1):
        img_name = source_img_path.stem + '_{}'.format(str(patch_id).zfill(3)) + source_img_path.suffix
        img_path = destination_dir.joinpath(img_name)
        cv2.imwrite(str(img_path), patch * 255.0)


def main(source_data_dir, destination_data_dir):
    device_num_patches_dict = {}
    devices = source_data_dir.glob("*")
    if not destination_data_dir.exists():
        os.makedirs(str(destination_data_dir), exist_ok=True)

    t_start = time.time()
    for device in devices:
        image_paths = device.glob("*")
        destination_device_dir = destination_data_dir.joinpath(device.name)
        
        # The following if-else construct makes sense on running multiple instances of this method 
        if destination_device_dir.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir), exist_ok=True)

        num_patches = 0
        for image_path in image_paths:
            # For now, we only want to extract frames from original videos
            if "WA" in image_path.stem or "YT" in image_path.stem:
                continue

            img = cv2.imread(str(image_path))
            img = np.float32(img) / 255.0

            patches = get_patches(img_data=img, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=20)
            num_patches += len(patches)
            save_patches(patches, image_path, destination_device_dir)

        device_num_patches_dict[device.name] = num_patches
        print(f"{device.name} | {num_patches} patches ({int(time.time() - t_start)} sec.)")

    return device_num_patches_dict

if __name__ == "__main__":
    images_per_device = Path('/data/s3523799/vbdi/datasets/Exp.III/balanced_ds_28D/train/')
    patches_per_device = Path('/data/s3523799/vbdi/datasets/Exp.III/patches_ds_28D/train/')
    device_patch_dict = main(images_per_device, patches_per_device)

    output_file_path = patches_per_device.joinpath("num_patches.txt")
    with open(output_file_path, "a") as text_file:
        print(f"{str(images_per_device)}", file=text_file)

        for key in device_patch_dict.keys():
            num_patches = device_patch_dict[key]
            print(f"{key}: {num_patches} patches extracted", file=text_file)
