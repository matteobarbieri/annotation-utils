#!/usr/bin/env python
# coding: utf-8

from pycocotools.coco import COCO
import numpy as np
import os

from PIL import Image

# from tqdm.notebook import tqdm
from tqdm import tqdm

import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'annotations_path', type=str, help="Path to the annotation json file.")

    parser.add_argument(
        'masks_path', type=str, help="Folders where to save segmentation masks.")

    return parser.parse_args()


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


def generate_mask_from_annotation(coco, anns, img_shape, cats, filterClasses):

    mask = np.zeros(img_shape, dtype='uint8')

    # Paint the annotations
    for i in range(len(anns)):

        className = getClassName(anns[i]['category_id'], cats)
        try:
            pixel_value = filterClasses.index(className)+1
        except ValueError as e:  # noqa
            # print('asdasdasd')
            # do not use that class
            continue

        mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)

    return mask


def generate_masks(coco, masks_folder, imgIds=None, filter_classes=None):
    """
    filter_classes : list
        The list of categories to be included in the segmentation mask.
        If None, use all categories included in the annotation file.
    """

    # Load the categories in a variable
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    # print(cats)
    cat_name_to_id = {x['name']: x['id'] for x in cats}
    cat_id_to_name = {x['id']: x['name'] for x in cats}

    filtered_cat_ids = list()
    # for c_name in ['Orange', 'Apple', 'Grape', 'Banana']:
    for c_name in filter_classes:
        filtered_cat_ids.append(cat_name_to_id[c_name])

    # Default classes to all classes, if not specified
    if filter_classes is None:
        filter_classes = [x['name'] for x in cats]

    # Get all images containing the above Category IDs
    if imgIds is None:
        imgIds = list()
        for c_id in filtered_cat_ids:
            imgIds.extend(coco.getImgIds(catIds=c_id))

        # Must set, since it contains duplicates
        imgIds = set(imgIds)

    for img_id in tqdm(imgIds):

        # Get the list of annotations ids for that image
        ann_ids = coco.getAnnIds([img_id])

        # Load the annotations from the list of annotation ids
        anns = coco.loadAnns(ann_ids)

        img_info = coco.imgs[img_id]

        # Save the image shape for future use
        img_shape = (img_info['height'], img_info['width'])

        # Also the original filename
        original_filename = img_info['file_name']

        # Actually generate the mask, returns a numpy array, dtype uint8
        mask = generate_mask_from_annotation(
            coco, anns, img_shape, cats, filter_classes)

        # Create the PIL version of the image, so that we can save it on disk
        pil_img = Image.fromarray(mask)

        # Save to disk (replace original extension with .png)
        mask_filename = os.path.join(masks_folder, f"{original_filename[:-4]}.png")

        pil_img.save(mask_filename)

    return imgIds


def main():

    args = parse_args()

    # annotation_file = "annotations/instances_annotations.json"

    # Initialize the COCO api for instance annotations
    coco = COCO(args.annotations_path)

    # Create folder where to store segmentation masks
    os.makedirs(args.masks_path, exist_ok=True)

    imgIds = generate_masks(  # noqa
        coco, args.masks_path,
        filter_classes=['Orange', 'Apple', 'Grape', 'Banana'])

    # ## OPTIONAL
    # copy only useful images

    # images_folder = 'images'

    # for img_id in tqdm(imgIds):
        # img_info = coco.imgs[img_id]

        # get_ipython().system("cp all_images/{img_info['file_name']} {images_folder}")


if __name__ == '__main__':
    main()
