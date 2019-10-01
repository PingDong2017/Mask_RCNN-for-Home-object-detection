import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

ROOT_DIR = os.path.abspath("../../")
HomeObject_DIR = os.path.join(ROOT_DIR, "datasets/home_object")
dataset_dir = os.path.join(HomeObject_DIR, 'test')

imgs = os.listdir(dataset_dir)

output_dir = os.path.join(HomeObject_DIR, 'test_new')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for img in imgs:
    head, ext = os.path.splitext(img)
    if ext not in ['.JPG','.jpg']:
        continue
    input =os.path.join(dataset_dir,img)
    image =cv2.imread(input)
    output=os.path.join(output_dir,img)
    cv2.imwrite(output,image)