# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:12:11 2019

@author: Koerber
"""


import imgaug.augmenters as iaa
import numpy as np


class ImageAugment():
    def __init__(self):
        self.outputwidth = 256
        self.outputheight = 256
        self.enableAll = True
        self.flip_horz = True
        self.flip_vert = True
        self.prob_gaussian = 0.4
        self.prob_piecewise = 0.3
        self.prob_dropout = 0.3
        
        self.enableAffine = True
        self.prob_affine = 0.5
        self.scale_min = 0.8
        self.scale_max = 1.2
        self.translate_min = -0.2
        self.translate_max = 0.2
        self.rotate_min = -45
        self.rotate_max = 45
        self.shear_min = -16
        self.shear_max = 16
        
        self.augmentation_sequence = None
        
    def initAugmentation(self):
        gaussian = lambda aug: iaa.Sometimes(self.prob_gaussian, aug)
        piecewise = lambda aug: iaa.Sometimes(self.prob_piecewise, aug)
        dropout = lambda aug: iaa.Sometimes(self.prob_dropout, aug)
        affine = lambda aug: iaa.Sometimes(self.prob_affine, aug)
        
        self.augmentation_sequence = iaa.Sequential([
            iaa.CropToFixedSize(width=self.outputwidth, height=self.outputheight),          
            iaa.PadToFixedSize(width=self.outputwidth, height=self.outputheight)])
        
        if not self.enableAll:
            return
        
        if self.flip_horz:
            print('flip horz')
            self.augmentation_sequence.append(iaa.Fliplr(0.5))
            
        if self.flip_vert:
            print('flip vert')
            self.augmentation_sequence.append(iaa.Flipud(0.5))
            
        if self.prob_gaussian > 0:
            print('flip gaussian')
            self.augmentation_sequence.append(gaussian(iaa.GaussianBlur(sigma=(0, 3.0))))
            
        if self.prob_piecewise > 0:
            print('flip piecewise')
            self.augmentation_sequence.append(piecewise(iaa.PiecewiseAffine(scale=(0.01, 0.05))))
            
        if self.prob_dropout > 0:
            print('flip dropout')
            self.augmentation_sequence.append(dropout(iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ])))
            
        if self.enableAffine and self.prob_affine > 0:
            print('flip affine')
            self.augmentation_sequence.append(affine(
                    iaa.KeepSizeByResize(
                    iaa.Affine(
                    scale={"x": (self.scale_min, self.scale_max), "y": (self.scale_min, self.scale_max)},
                    translate_percent={"x": (self.translate_min, self.translate_max), "y": (self.translate_min, self.translate_max)},
                    rotate=(self.rotate_min, self.rotate_max), 
                    shear=(self.shear_min, self.shear_max), 
                    order=[0, 1], 
                    mode=["reflect"] # what is best padding strategy for segmentation ?!
                ))))
            
        print(self.augmentation_sequence)
            
    def augment(self, image, label):
        x,y = self.augmentation_sequence(images=image, segmentation_maps=label)
        return np.asarray(x),np.asarray(y)


# OUTPUTSIZE = 256    # this size determines the training image size

# sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# rare = lambda aug: iaa.Sometimes(0.2, aug)
# seq = iaa.Sequential([
#     iaa.CropToFixedSize(width=OUTPUTSIZE, height=OUTPUTSIZE),          
#     iaa.PadToFixedSize(width=OUTPUTSIZE, height=OUTPUTSIZE),  
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), 
#     sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
#     sometimes(
#             iaa.KeepSizeByResize(
#             iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, 
#             rotate=(-45, 45), 
#             shear=(-16, 16), 
#             order=[0, 1], 
#             mode=["constant"] 
#         ))),
#     rare(iaa.OneOf([
#                     iaa.Dropout((0.01, 0.1), per_channel=0.5),
#                     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
#                 ]))
# ])

# def augment(image, label): 
#     x,y = seq(images=image, segmentation_maps=label)
#     return np.asarray(x),np.asarray(y)
