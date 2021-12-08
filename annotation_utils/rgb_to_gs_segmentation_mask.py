#!/usr/bin/env python
# coding: utf-8

import os

from glob import glob

from PIL import Image

import numpy as np

from tqdm import tqdm

import sys

# DATA_FOLDER =
# '/mnt/storage/data/object_detection/helmet_100/annotated/segmentation_masks/validation'  # noqa


def create_label_map(label_map_file: str, background_class='background'):
    # Create map from label to RGB code

    label_rgb_map = dict()

    label_idx = 1

    with open(label_map_file, 'r') as lf:

        # Skip first line, has the header
        lf.readline()

        line = lf.readline().strip()

        while line:

            line_arr = line.split(':')

            if line_arr[0] != background_class:

                # Save reference color and index value
                label_rgb_map[line_arr[0]] = (
                    np.array(
                        [int(x) for x in line_arr[1].split(',')]),
                    label_idx
                )

                label_idx += 1

            # Read next line
            line = lf.readline().strip()

    return label_rgb_map


def rgb_to_gs_mask(rgb_im_array: np.array, label_rgb_map: dict) -> np.array:

    # Create blank mask
    gs_mask = np.zeros(rgb_im_array.shape[:2], dtype='uint8')

    for _, (c, idx) in label_rgb_map.items():

        # Add index-label for each pixel
        gs_mask += np.all(rgb_im_array == c, axis=-1).astype('uint8') * idx

    return gs_mask


def main():

    DATA_FOLDER = sys.argv[1]

    LABEL_MAP_FILE = os.path.join(DATA_FOLDER, 'labelmap.txt')
    RGB_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'SegmentationClass')
    GS_MASKS_FOLDER = os.path.join(DATA_FOLDER, 'SegmentationClassRaw')

    rgb_masks_list = glob(RGB_MASKS_FOLDER + "/*")

    label_rgb_map = create_label_map(LABEL_MAP_FILE)

    for rgb_file in tqdm(rgb_masks_list):

        # Load original rgb image
        im_array = np.array(Image.open(rgb_file))

        # Actually generate the gs mask (numpy array)
        gs_mask_array = rgb_to_gs_mask(im_array, label_rgb_map)

        # Generate destination file name: must have same file name as original file
        gs_filename = os.path.join(GS_MASKS_FOLDER, os.path.basename(rgb_file))

        # Save file to disk
        Image.fromarray(gs_mask_array).save(gs_filename, 'PNG')


if __name__ == '__main__':

    main()