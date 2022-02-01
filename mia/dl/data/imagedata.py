# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:23 2020

@author: Koerber
"""

import os
import glob
import cv2
from dl.data.labels import getMatchingImageLabelPairsRecursive
import numpy as np
from utils.Image import ImageFile
import random
import concurrent.futures
from itertools import repeat
import math
import random
import time


class ImageData():
    def __init__(self, parent):
        self.parent = parent
        
        # should we split validation data and image data in 2 dataset classes ?
        self.TrainingImagePaths = []
        self.TrainingLabelPaths = []
        self.TrainingTileIndices = []
        
        # TrainTestSplit = 1 means 100% Training Data and 0% validation
        self.TrainTestSplit = 0.85
        self.UseValidationDir = False
        
        self.IgnoreBorder = 0
        
        self.autocalcClassWeights = False
        self.class_weights = None 
        self.classValues = None
        
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
    
    def clearPaths(self):
        self.TrainingImagePaths.clear()
        self.TrainingLabelPaths.clear()
        self.TrainingTileIndices.clear()
        self.ValidationImagePaths.clear()
        self.ValidationLabelPaths.clear()
        self.ValidationTileIndices.clear()
    
            
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
        image = file.getDLInputImage(self.parent.MonoChrome,frame)
        return image
    
    
    def initTrainingDataset(self, train_imagepath, val_imagepath):
        self.clearPaths()
        images, labels = getMatchingImageLabelPairsRecursive(train_imagepath, self.parent.LabelFolderName)
        if images is None or labels is None or not images or not labels:
            return None, None
        
        
        # careful: we got new validation set each time we start training
        z = list(zip(images, labels))

        random.shuffle(z)
        images, labels = zip(*z)
        # why does zip return tuple?
        images = list(images)
        labels = list(labels)

        if self.UseValidationDir:  
            self.TrainingImagePaths = images
            self.TrainingLabelPaths = labels
            
            if val_imagepath and os.path.isdir(val_imagepath):
                val_images, val_labels = getMatchingImageLabelPairsRecursive(val_imagepath, self.parent.LabelFolderName)
                self.ValidationImagePaths = val_images
                self.ValidationLabelPaths = val_labels
        else:
            split = int(self.TrainTestSplit*len(images))
    
            if split > 0:
                self.TrainingImagePaths = images[0:split]
                self.TrainingLabelPaths = labels[0:split]
                self.ValidationImagePaths = images[split:]
                self.ValidationLabelPaths = labels[split:]
            else:
                self.TrainingImagePaths = images
                self.TrainingLabelPaths = labels


    def initialized(self):
        return True if len(self.TrainingImagePaths) * len(self.TrainingLabelPaths) > 0 else False
    
    def validationData(self):
        return True if len(self.ValidationImagePaths) * len(self.ValidationLabelPaths) > 0 else False

    def preprocessImage(self, image):
        return self.parent.Mode.preprocessImage(image)
        
    def preprocessLabel(self,label):
        return self.parent.Mode.preprocessLabel(label)
    
    
    def removeUnlabelledData(self, image, label):
        _, thresh = cv2.threshold(label, 0, 255,  cv2.THRESH_BINARY)
        image = cv2.bitwise_and(image,image, mask=thresh)
            
        # remove background
        label[np.where(label == 255)] = 0
        return image, label

            
    def createImageLabelPair(self, index, validation=False):
        images = self.getImagePaths(validation)
        masks = self.getLabelPaths(validation)

        channels = 1 if self.parent.MonoChrome is True else 3

        # handle image stacks and others
        # using same label for complete stack unhandled atm
        if isinstance(masks[index], tuple):
            mask = masks[index][0]
            try:
                frame = int(masks[index][1])
            except:
                print(masks[index][1])
                raise # not a valid label for a stack
            image = images[index][0]
        else:
            mask = masks[index]
            frame = None
            image = images[index]
            
        train_img = self.readImage(image, frame)

        width, height = self.parent.Mode.getImageSize4ModelInput(train_img.shape[1], train_img.shape[0])
              
        train_mask = self.parent.Mode.LoadLabel(mask, train_img.shape[0], train_img.shape[1])

        train_img, train_mask = self.removeUnlabelledData(train_img, train_mask) 
        train_img =  cv2.resize(train_img, (width, height))
        train_mask = self.parent.Mode.resizeLabel(train_mask, (width, height))

        if self.autocalcClassWeights and not validation:
            if self.classValues is not None:
                self.classValues += self.parent.Mode.countLabelWeight(train_mask)

        train_mask = train_mask.reshape(height, width, 1)
        train_img = train_img.reshape(height, width, channels)
        
        if self.IgnoreBorder > 0 and 3*self.IgnoreBorder < height and 3*self.IgnoreBorder < width:
            train_mask = train_mask[self.IgnoreBorder:height-2*self.IgnoreBorder,self.IgnoreBorder:width-2*self.IgnoreBorder,:].copy()
            train_img = train_img[self.IgnoreBorder:height-2*self.IgnoreBorder,self.IgnoreBorder:width-2*self.IgnoreBorder,:].copy()
        

        train_mask = self.parent.Mode.prepreprocessLabel(train_mask)
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
            for _ in range(xtimes):
                index = random.randint(0,numImages-1)
                
                if isinstance(images[index], tuple):
                    image = images[index][0]
                else:
                    image = images[index]
                img = self.readImage(image)
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
        if not validation:
            self.initclassValues()

        im_channels = 1 if self.parent.MonoChrome is True else 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.worker) as executor:
            x = executor.map(self.createImageLabelPair,range(numImages), repeat(validation))
        if not validation:
            self.calcClassWeights()
        img = []
        mask = []

        for counter, pair in enumerate(x): 
            tiles = [counter] * self.parent.augmentation.getTilesPerImage(pair[0].shape[1],pair[0].shape[0])
            self.addTileIndices(tiles, validation)
            img.append(pair[0])
            mask.append(pair[1])
        return img, mask

    def getAutoClassWeights(self):
        # this is only used when not training in memory
        # up to 1000 images of training set are loaded and classweights calculated
        # would be much faster to only load labels 
        if not self.initialized():
            return
        self.initclassValues()
        if self.autocalcClassWeights:
            numImages = len(self.getImagePaths(False))
            for i in range(min(1000,numImages)):
                self.createImageLabelPair(i)
                
        self.calcClassWeights()
        
    def setClassWeights(self, weights):
        self.class_weights = weights
        
    def resetClassWeights(self):
        self.class_weights = None

    def initclassValues(self):
        self.classValues = np.zeros((self.parent.NumClasses_real), dtype = np.float)

    def calcClassWeights(self):
        if self.autocalcClassWeights:
            self.class_weights = np.sum(self.classValues)/self.classValues
            self.class_weights /= np.sum(self.class_weights)
            if self.parent.NumClasses_real == 2:
                self.class_weights = self.class_weights[1]
