# Mask_RCNN-for-Home-object-detection

This is an example to show the use of Mask RCNN in home object detection. 


## Train the model

Train a new model starting from pre-trained COCO weights
```
python home_objct.py train --dataset=/path/to/home_object/dataset --weight=cocoo
```

##Object detection and color splash
```
python home_object.py splash --weights=/path/to/mask_rncc_homeobject.h5 --image=<file name>
```
