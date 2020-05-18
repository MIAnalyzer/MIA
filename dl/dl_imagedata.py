# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:23 2020

@author: Koerber
"""

import os
import glob
import cv2
import dl.dl_utils as dl_utils
from utils.Contour import LoadLabel
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.Image import normalizeImage

import concurrent.futures
from itertools import repeat

class ImageData():
    def __init__(self, parent):
        self.parent = parent
        self.TrainingImagePath = None
        self.LabelImagePath = None
        
        self.ImagesPaths = []
        self.LabelsPaths = []

        
    def readImage(self, path):
        
        # this should be changed to ImageFile class and should also handle stacks
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # should we normalize each image or over the dataset?
        # pro dataset: better for images from the same source, i.e. images that do not contain any target will lead to background that gets amplified by single image normalization
        # con dataset: single image normalization is better suited for different sources, like 8-bit and 16-bit and color images combined
        image = normalizeImage(image)

        if len(image.shape) == 4:
            image = cv2.cvtColor(image, cv2.CV_BGRA2BGR )
        if len(image.shape) == 2:
            if not self.parent.MonoChrome():
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
            else:
                image = image[..., np.newaxis]
        elif len(image.shape) == 3:
            if self.parent.MonoChrome() and image.shape[2] > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
                # atm I dont know if bgr2gray squeezes
                if len(image.shape) == 2:
                    image = image[..., np.newaxis]    
        return image
    
    
    def initTrainingDataset(self, imagepath, labelpath):         
        images = glob.glob(os.path.join(imagepath,'*.*'))
        labels = glob.glob(os.path.join(labelpath,'*.*'))
          
        # to do: check if images in one of supported image formats
        labels = [x for x in labels if x.endswith(".npy") or x.endswith(".npz")]
        
        image_names = [os.path.splitext(os.path.basename(each))[0] for each in images]
        label_names = [os.path.splitext(os.path.basename(each))[0] for each in labels]
        
        intersection = list(set(image_names).intersection(label_names))
        if intersection == list():
            return None, None
        
        images = [each for each in images if os.path.splitext(os.path.basename(each))[0] in intersection]
        labels = [each for each in labels if os.path.splitext(os.path.basename(each))[0] in intersection]
        images.sort()
        labels.sort()
        
        self.ImagesPaths = images
        self.LabelsPaths = labels

    def initialized(self):
        return True if len(self.ImagesPaths) * len(self.LabelsPaths) > 0 else False

    def preprocessImage(self, image):
        return image.astype('float')/255.
        
    def preprocessLabel(self,mask):
        if self.parent.NumClasses() > 2:
            if self.parent.useWeightedDistanceMap:                 
                label = mask[...,0]
                weights = mask[...,1]
                label = to_categorical(label, num_classes=self.parent.NumClasses())
                mask = np.concatenate((label, weights[...,np.newaxis]), axis = 3) 
            else:
                mask = to_categorical(mask, num_classes=self.parent.NumClasses())
        else:
            mask = mask.astype('float')
        return mask
    
    
    def removeUnlabelledData(self, image, label):
        # remove unlabelled data
        _, thresh = cv2.threshold(label, 0, 255,  cv2.THRESH_BINARY)
        image = cv2.bitwise_and(image,image, mask=thresh)
            
        # remove background
        label[np.where(label == 255)] = 0
        return image, label


    def addWeightMap(self, label):
        if self.parent.useWeightedDistanceMap:
            weights = dl_utils.createWeightedBorderMapFromLabel(label[np.newaxis,...])
            label = np.concatenate((label, weights[0,...]), axis = 2)
        return label
            
    def createImageLabelPair(self, index, width=None, height=None):

        channels = 1 if self.parent.MonoChrome() is True else 3
        train_img = self.readImage(self.ImagesPaths[index])
        
        if not width or not height:
            width = int(train_img.shape[1]*self.parent.ImageScaleFactor)
            height= int(train_img.shape[0]*self.parent.ImageScaleFactor)
                
        train_mask = LoadLabel(self.LabelsPaths[index], train_img.shape[0], train_img.shape[1])
            
        train_img =  cv2.resize(train_img, (width, height))
        train_mask = cv2.resize(train_mask, (width, height), interpolation = cv2.INTER_NEAREST )

        train_img, train_mask = self.removeUnlabelledData(train_img, train_mask) 

        train_mask = train_mask.reshape(height, width, 1)
        train_img = train_img.reshape(height, width, channels)
            
        train_mask = self.addWeightMap(train_mask)
        
        return train_img, train_mask
    

    
    def loadTrainingDataSet(self):
        if not self.initialized():
            return
       
        images = self.ImagesPaths
        test = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
        width = int(test.shape[1]*self.parent.ImageScaleFactor)
        height = int(test.shape[0]*self.parent.ImageScaleFactor)
        
        
        im_channels = 1 if self.parent.MonoChrome() is True else 3
        lb_channels = 2 if self.parent.useWeightedDistanceMap else 1
        # now: all images are resized to the size of the first image 
        # Alternatively: all images could be padded to the largest image size
        img = np.zeros((len(images), height, width, im_channels)).astype('uint8')
        mask = np.zeros((len(images), height, width, lb_channels)).astype('uint8')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.worker) as executor:
            x = executor.map(self.createImageLabelPair,range(len(images)), repeat(width), repeat(height))
        
        for counter, pair in enumerate(x):   
            img[counter] = pair[0]
            mask[counter] = pair[1]
        
        return img, mask
    
    
    

