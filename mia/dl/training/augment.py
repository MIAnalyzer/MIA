# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:12:11 2019

@author: Koerber
"""


import imgaug.augmenters as iaa
import numpy as np
import random
import math
import cv2
from dl.method.mode import dlMode

class ImageAugment():
    def __init__(self, parent):
        # this is the model input size
        self.parent = parent
        self.modelinputwidth = 224  
        self.modelinputheight = 224
        
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
        
        self.seg_ignoreBackground = True
        self.seg_ignorecycles = 5
        self.seg_minRequiredPixels = 1
        
        self.od_ignoreBackground = True
        self.od_ignorecycles = 5
        self.od_minRequiredPixels = 1
        
        self.augmentation_sequence = None
        
        self.currentTile = 0


    @property
    def outputwidth(self):
        # it is called outputwidth because it is the output of the augmentation and the input to the model
        try:
            if self.parent.InputSize[1] is not None:
                return self.parent.InputSize[1]
        except:
            pass
        return self.modelinputwidth

    @outputwidth.setter
    def outputwidth(self, val):
        self.modelinputwidth = val


    @property
    def outputheight(self):
        # it is called outputwidth because it is the output of the augmentation and the input to the model
        try:
            if self.parent.InputSize[0] is not None:
                return self.parent.InputSize[0]
        except:
            pass
        return self.modelinputheight

    @outputheight.setter
    def outputheight(self, val):
        self.modelinputheight = val
        
    @property
    def ignoreBackground(self):
        if self.parent.WorkingMode == dlMode.Segmentation:
            return self.seg_ignoreBackground
        elif self.parent.WorkingMode == dlMode.Object_Counting:
            return self.od_ignoreBackground
        else:
            return False
        
    @property
    def ignorecycles(self):
        if self.parent.WorkingMode == dlMode.Segmentation:
            return self.seg_ignorecycles
        elif self.parent.WorkingMode == dlMode.Object_Counting:
            return self.od_ignorecycles
        else:
            return 0
    
    @property
    def minRequiredPixels(self):
        if self.parent.WorkingMode == dlMode.Segmentation:
            return self.seg_minRequiredPixels
        elif self.parent.WorkingMode == dlMode.Object_Counting:
            return self.od_minRequiredPixels
        else:
            return 1
        
    def setoutputwidth(self, w):
        self.outputwidth = w - w % self.parent.split_factor
        
    def setoutputheight(self, h):
        self.outputheight = h - h % self.parent.split_factor
        
    def randomcrop_or_pad(self, images, labels):
        # input a list of images
        # output a single array with images (num, outputwidth, outputheight)
        imgs = []
        labls = []
        for image,label in zip(images, labels):
            assert (label.shape[0:1] == image.shape[0:1])
            if image.shape[1] < self.outputwidth or image.shape[0] < self.outputheight:
                w = max(0,self.outputwidth - image.shape[1])
                h = max(0,self.outputheight - image.shape[0])
                # after padding size is increased by h or w in that direction
                pad = ((h, h), (w, w), (0, 0))
                image = np.pad(image, pad, 'reflect')
                label = np.pad(label, pad, 'reflect')

            if self.ignoreBackground:
                counter = 0
                while counter < self.ignorecycles:
                    x = random.randint(0, image.shape[1] - self.outputwidth)
                    y = random.randint(0, image.shape[0] - self.outputheight)
                    if np.count_nonzero(label[y:y+self.outputheight, x:x+self.outputwidth,...]) >= self.minRequiredPixels:
                        break
                    counter += 1
            else:
                x = random.randint(0, image.shape[1] - self.outputwidth)
                y = random.randint(0, image.shape[0] - self.outputheight)
                
            image = image[y:y+self.outputheight, x:x+self.outputwidth,...]
            label = label[y:y+self.outputheight, x:x+self.outputwidth,...]
 
            imgs.append(image)
            labls.append(label)
            
        return imgs, labls
    
    def getTilesPerImage(self, width, height):
        if self.parent.Mode.type == dlMode.Segmentation or self.parent.Mode.type == dlMode.Object_Counting:
            return  math.ceil(width/self.outputwidth) * math.ceil(height/self.outputheight)
        elif self.parent.Mode.type == dlMode.Classification:
            return 1
            
    
    def systematiccrop_or_pad(self, images, labels):     
        # images are cropped from left-to-right and top-to-bot      
        # self.currentTile keeps track of the position in the parent image 
        # it requires sorted images as input to return correct tiles
        imgs = []
        labls = []
        for image,label in zip(images, labels):           
            assert (label.shape[0:2] == image.shape[0:2])
            if image.shape[1] < self.outputwidth or image.shape[0] < self.outputheight:
                w = max(0,self.outputwidth - image.shape[1])
                h = max(0,self.outputheight - image.shape[0])
                # after padding size is increased by h or w in that direction
                pad = ((h, 0), (w, 0), (0, 0))
                image = np.pad(image, pad, 'reflect')
                label = np.pad(label, pad, 'reflect')

            
            num_in_width = math.ceil(image.shape[1] / self.outputwidth)
            num_in_height = math.ceil(image.shape[0] / self.outputheight)
            x = (self.currentTile % num_in_width) * self.outputwidth
            y = (self.currentTile // num_in_width) * self.outputheight
            
            if self.currentTile+1 >= self.getTilesPerImage(image.shape[1],image.shape[0]):
                self.currentTile = 0
            else:
                self.currentTile += 1
                
            x_start = min(x, image.shape[1] - self.outputwidth)
            y_start = min(y, image.shape[0] - self.outputheight)
            x_end = min(x+self.outputwidth, image.shape[1])
            y_end = min(y+self.outputheight, image.shape[0])
            image = image[y_start:y_end, x_start:x_end,...]
            label = label[y_start:y_end, x_start:x_end,...]
            
            
            imgs.append(image)
            labls.append(label)

        return imgs, labls
        
    def fitFullSizeImage2OutputSize(self, images):
        # add random crop?
        dims = images[0].shape
        ret = [cv2.resize(x, (self.outputwidth, self.outputheight), interpolation = cv2.INTER_CUBIC) for x in images]
        # in case of monochromatic
        if dims > ret[0].shape:
            ret = [x[...,np.newaxis] for x in ret]
        return ret

    def checkModelSizeCompatibility(self, image, label  = None):       
        targetsize = self.parent.InputSize
        currsize = image.shape[1:3]
        idims = len(image.shape)
        
        if targetsize[0] is None and targetsize[1] is None:
            targetsize = (self.outputheight, self.outputwidth)
        
        
        if (targetsize[0] is not None or targetsize[1] is not None) and (targetsize != currsize):
            image = np.stack([cv2.resize(x, (targetsize[1], targetsize[0]), interpolation = cv2.INTER_CUBIC) for x in image], axis=0)
            if len(image.shape) < idims:
                image = image[...,np.newaxis]
            if label is not None:
                ldims = len(label.shape)
                label = np.stack([cv2.resize(x, (targetsize[1], targetsize[0]), interpolation = cv2.INTER_NEAREST) for x in label], axis=0)
                if len(label.shape) < ldims:
                    label = label[...,np.newaxis]
        if label is not None:
            return image, label
        return image
        
    def initAugmentation(self):
        # imgaug was compared to albumentation and imgaug was faster (see test sheet)
        # problem with albumentation it only takes a single image as input and processes 1-by-1

        self.currentTile = 0
        gaussian = lambda aug: iaa.Sometimes(self.prob_gaussian, aug)
        piecewise = lambda aug: iaa.Sometimes(self.prob_piecewise, aug)
        dropout = lambda aug: iaa.Sometimes(self.prob_dropout, aug)
        affine = lambda aug: iaa.Sometimes(self.prob_affine, aug)
        

        self.augmentation_sequence = iaa.Sequential()
        
        if not self.enableAll:
            return
        
        if self.flip_horz:
            self.augmentation_sequence.append(iaa.Fliplr(0.5))
            
        if self.flip_vert:
            self.augmentation_sequence.append(iaa.Flipud(0.5))
            
        if self.prob_gaussian > 0:
            self.augmentation_sequence.append(gaussian(iaa.GaussianBlur(sigma=(0, 3.0))))
            
        if self.prob_piecewise > 0:
            self.augmentation_sequence.append(piecewise(iaa.PiecewiseAffine(scale=(0.01, 0.05))))
            
        if self.prob_dropout > 0:
            self.augmentation_sequence.append(dropout(iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ])))
            
        if self.enableAffine and self.prob_affine > 0:
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

            
    def augment(self, image, label, validation = False):     
        if self.parent.Mode.type == dlMode.Segmentation or self.parent.Mode.type == dlMode.Object_Counting:
            if validation:
                x, y = self.systematiccrop_or_pad(image, label)
            else:
                image, label = self.randomcrop_or_pad(image, label)
                x,y = self.augmentation_sequence(images=image, segmentation_maps=label)
            x, y = self.checkModelSizeCompatibility(np.asarray(x), np.asarray(y))

        elif self.parent.Mode.type == dlMode.Classification:
            if validation:
                x = self.fitFullSizeImage2OutputSize(image)
                y = label
            else:
                image = self.fitFullSizeImage2OutputSize(image)
                y = label
                x = self.augmentation_sequence(images=image)
            
            x = self.checkModelSizeCompatibility(np.asarray(x))
            y = np.asarray(y)
        else:
            raise 
        return x,y

