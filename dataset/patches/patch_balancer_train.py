import os
import random
import shutil
from pathlib import Path


def balance_patches(unbalanced_dir, balanced_dir):
    # Remove old directories
    if balanced_dir.exists():
        shutil.rmtree(balanced_dir)

    # Construct a hierarchical dictionary
    # device_names
    #   |-- image names
    #        |-- patches names
    patches_dictionary = {}
    for device in unbalanced_dir.glob('*'):
        if not device.is_dir():
            continue

        patches_dictionary[device.name] = {}
        for patch in device.glob('*'):
            image_name = '_'.join(patch.name.split('_')[:-1])
            if image_name not in patches_dictionary[device.name]:
                patches_dictionary[device.name][image_name] = [patch]
            else:
                patches_dictionary[device.name][image_name].append(patch)
        print(f"{device.name} | num frames: {len(patches_dictionary[device.name])}")

    # determine the minimum num of images per device
    min_num_samples = float('inf')
    min_num_device = None
    for device_name in patches_dictionary:
        num_samples = len(patches_dictionary[device_name])
        if num_samples < min_num_samples:
            min_num_samples = num_samples
            min_num_device = device_name

    print(f"Device {min_num_device} has lowest number of frames: {min_num_samples}")

    # Create the directory structure along with symbolic links
    for device_name in patches_dictionary:
        # create device subdir in the destination folder
        subdir = balanced_dir.joinpath(device_name)
        Path(subdir).mkdir(parents=True, exist_ok=True)

        # randomly select min_num_images
        images = list(patches_dictionary[device_name].keys())
        random.seed(123)  # fixed seed to produce reproducible results
        random.shuffle(images)
        images = images[:min_num_samples]

        # create symlinks
        for image in images:
            for patch_path in patches_dictionary[device_name][image]:
                symlink = subdir.joinpath(patch_path.name)
                if not symlink.exists():
                    os.symlink(src=patch_path, dst=symlink)

        print(f"{device_name} Finished balancing")


if __name__ == "__main__":
    unbalanced = Path('/data/s3523799/vbdi/datasets/Exp.III/patches_ds_28D/train/')
    balanced = Path('/data/s3523799/vbdi/datasets/Exp.III/balanced_patches_ds_28D/train/')

    balance_patches(unbalanced_dir=unbalanced, balanced_dir=balanced)


