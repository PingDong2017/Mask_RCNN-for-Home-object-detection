
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

ROOT_DIR = os.path.abspath("../../")
HomeObject_DIR = os.path.join(ROOT_DIR, "datasets/home_object")
dataset_dir = os.path.join(HomeObject_DIR, 'train')

annotations = json.load(open(os.path.join(dataset_dir, "via_regions.json")))
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations = [a for a in annotations if a['regions']]

for a in annotations:
    # Get the x, y coordinaets of points of the polygons that make up
    # the outline of each object instance. These are stores in the
    # shape_attributes (see json format above)
    # The if condition is needed to support VIA versions 1.x and 2.x.
    if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in a['regions']]

    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    height, width = image.shape[:2]

    for i in range(len(polygons)):
        if max(polygons[i]['all_points_x'])>=width or max(polygons[i]['all_points_y'])>= height:
            print(f"image:{a['filename']}")

print('done')