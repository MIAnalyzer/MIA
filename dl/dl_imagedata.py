# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:23 2020

@author: Koerber
"""

import os
import glob
import cv2
import dl.dl_utils as dl_utils
from dl.dl_labels import LoadLabel, getAllImageLabelPairPaths, splitStackLabels
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.Image import ImageFile
import random
import concurrent.futures
from itertools import repeat, chain
import math
import random

class ImageData():
    def __init__(self, parent):
        self.parent = parent
        
        # should we split validation data and image data in 2 dataset classes ?
        self.TrainingImagePaths = []
        self.TrainingLabelPaths = []
        self.TrainingTileIndices = []
        
        # TrainTestSplit = 1 means 100% Training Data and 0% validation
        self.TrainTestSplit = 0.85
        
        self.ValidationImagePaths = []
        self.ValidationLabelPaths = []
        self.ValidationTileIndices = []

    def getImagePaths(self, validation):
        if validation:
            return self.ValidationImagePaths
        else:
            return self.TrainingImagePaths
            
        
    def getLabelPaths(self, validation):
        if validation:
            return self.ValidationLabelPaths
        else:
            return self.TrainingLabelPaths
    
            
    def addTileIndices(self, tiles, validation):
        if validation:
            self.ValidationTileIndices.extend(tiles)
        else:
            self.TrainingTileIndices.extend(tiles)
        
    def readImage(self, path, frame = 0):

        file = ImageFile(path)
        # should we normalize each image or over the dataset?
        # pro dataset: better for images from the same source, i.e. images that do not contain any target will lead to background that gets amplified by single image normalization
        # con dataset: single image normalization is better suited for different sources, like 8-bit and 16-bit and color images combined
        file.normalizeImage()
        image = file.getDLInputImage(self.parent.MonoChrome(),frame)
        return image
    
    
    def initTrainingDataset(self, imagepath, labelpath):   
        images, labels = getAllImageLabelPairPaths(imagepath, labelpath)
        if images is None or labels is None:
            return None, None
        
        split = [splitStackLabels(x,y) for x,y in zip(images, labels)]
        images,labels = zip(*split)
        images = list(chain.from_iterable(images))
        labels = list(chain.from_iterable(labels))

        # careful: we got new validation set each time we start training
        z = list(zip(images, labels))

        random.shuffle(z)
        images, labels = zip(*z)
        
        split = int(self.TrainTestSplit*len(images))
        self.TrainingImagePaths = images[0:split]
        self.TrainingLabelPaths = labels[0:split]
        self.TrainingTileIndices = []
        
        self.ValidationImagePaths = images[split:]
        self.ValidationLabelPaths = labels[split:]
        self.ValidationTileIndices = []


    def initialized(self):
        return True if len(self.TrainingImagePaths) * len(self.TrainingLabelPaths) > 0 else False
    
    def validationData(self):
        return True if len(self.ValidationImagePaths) * len(self.ValidationLabelPaths) > 0 else False

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
            
    def createImageLabelPair(self, index, validation=False):

        images = self.getImagePaths(validation)
        masks = self.getLabelPaths(validation)

        channels = 1 if self.parent.MonoChrome() is True else 3

        # handle image stacks and others
        if isinstance(masks[index], tuple):
            mask = masks[index][0]
            frame = int(masks[index][1])
            image = images[index][0]
        else:
            mask = masks[index]
            frame = None
            image = images[index]

        train_img = self.readImage(image, frame)
        
        width = int(train_img.shape[1]*self.parent.ImageScaleFactor)
        height= int(train_img.shape[0]*self.parent.ImageScaleFactor)
              
        train_mask = LoadLabel(mask, train_img.shape[0], train_img.shape[1])
            
        train_img =  cv2.resize(train_img, (width, height))
        train_mask = cv2.resize(train_mask, (width, height), interpolation = cv2.INTER_NEAREST )

        train_img, train_mask = self.removeUnlabelledData(train_img, train_mask) 

        train_mask = train_mask.reshape(height, width, 1)
        train_img = train_img.reshape(height, width, channels)
            
        train_mask = self.addWeightMap(train_mask)
        
        return train_img, train_mask
    
    def getTileIndices(self, validation = False, equalTilesperImage = False):
        if not equalTilesperImage:
            if validation:
                self.ValidationTileIndices.sort()
                return self.ValidationTileIndices
            else:
                return self.TrainingTileIndices
        else:
            images = self.getImagePaths(validation)
            numImages = len(images)
            # draw xtimes an images and calculate tilenumber and equally distribute
            xtimes = 5
            tilenum = 0
            for _ in range(5):
                index = random.randint(0,numImages-1)
                img = self.readImage(images[index])
                width = int(img.shape[1]*self.parent.ImageScaleFactor)
                height= int(img.shape[0]*self.parent.ImageScaleFactor)
                tilenum += self.parent.augmentation.getTilesPerImage(width, height)
            tilenum = tilenum // xtimes
            ret = []
            [ret.extend([i] * tilenum) for i in range(numImages)]
            if validation:
                ret.sort()
                self.ValidationTileIndices = ret
            else:
                self.TrainingTileIndices = ret
            return ret
        
    def getNumberOfBatches(self, validation = False):
        if validation:
            idc = len(self.ValidationTileIndices)
        else:
            idc = len(self.TrainingTileIndices)
            
        return int(np.ceil(idc / float(self.parent.batch_size)))
                
    
    def loadTrainingDataSet(self, validation = False):
        if not self.initialized():
            return None, None
        
        images = self.getImagePaths(validation)
        numImages = len(images)

        
        im_channels = 1 if self.parent.MonoChrome() is True else 3
        lb_channels = 2 if self.parent.useWeightedDistanceMap else 1
      
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.worker) as executor:
            x = executor.map(self.createImageLabelPair,range(numImages), repeat(validation))
        
        img = []
        mask = []

        for counter, pair in enumerate(x): 
            tiles = [counter] * self.parent.augmentation.getTilesPerImage(pair[0].shape[1],pair[0].shape[0])
            self.addTileIndices(tiles, validation)
            img.append(pair[0])
            mask.append(pair[1])
        return img, mask

