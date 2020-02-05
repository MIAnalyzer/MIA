# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:12:11 2019

@author: Koerber
"""


import imgaug.augmenters as iaa
import numpy as np


OUTPUTSIZE = 256    # this size determines the training image size

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
rare = lambda aug: iaa.Sometimes(0.2, aug)
seq = iaa.Sequential([
    iaa.CropToFixedSize(width=OUTPUTSIZE, height=OUTPUTSIZE),          
    iaa.PadToFixedSize(width=OUTPUTSIZE, height=OUTPUTSIZE),  
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), 
    sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
    sometimes(
            iaa.KeepSizeByResize(
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
            rotate=(-45, 45), 
            shear=(-16, 16), 
            order=[0, 1], 
            mode=["constant"] 
        ))),
    rare(iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]))
])

def augment(image, label):
    x,y = seq(images=image, segmentation_maps=label)
    #seq.show_grid([x[0], y[0]*255], cols=8, rows=1)
    return np.asarray(x),np.asarray(y)
    
    
