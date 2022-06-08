# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:47:31 2019

@author: Koerber
"""



import cv2
import numpy as np
import random
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.python.keras.utils.data_utils import Sequence
import dl.utils.dl_utils as dl_utils

class TrainingDataGenerator(Sequence):
    def __init__(self, parent, validation = False):
        self.parent = parent
        self.lock = threading.Lock()
        self.inMemory  = parent.TrainInMemory
        self.validation = validation
        if self.inMemory:
            self.images,self.labels = parent.data.loadTrainingDataSet(validation = self.validation)
        else:
            self.images = self.parent.data.getImagePaths(self.validation)
            self.labels = self.parent.data.getLabelPaths(self.validation)
            self.parent.data.getAutoClassWeights()
            
        self.parent.augmentation.initAugmentation()
        self.indices = self.parent.data.getTileIndices(validation, equalTilesperImage = not self.inMemory)
        # self.numImages = len(self.images)
        # experimental: 
        # calculate the number of tiles as the possible crops from all images
        # note that crops may overlap (avoided for validation)
        # length of generator derived from tiles
        self.numTiles = len(self.indices)
        if not validation:
            random.shuffle(self.indices)

    def __len__(self):
        # return int(np.ceil(self.numImages / float(self.batch_size)))
        return self.parent.data.getNumberOfBatches(self.validation)

    def __getitem__(self, idx):
        # important question: do we need a lock here (slows down training time significantly)
        # so far no problems detected
        #with self.lock:

            batch_start = idx * self.parent.batch_size
            batch_end = min(batch_start + self.parent.batch_size, self.numTiles)
            if self.inMemory:
                batch_images,batch_masks = self._loadBatchFromMemory(batch_start, batch_end)
            else:
                batch_images,batch_masks = self._loadBatchFromDisk(batch_start, batch_end)
            
            img, mask = self.parent.augmentation.augment(batch_images,batch_masks, self.validation)

            img = self.parent.data.preprocessImage(img)   
            mask = self.parent.data.preprocessLabel(mask)                     
            return img, mask
        
    def _loadBatchFromMemory(self, batch_start, batch_end): 
        # are we sure this create a shallow copy? otherwise it would be terrible inefficient memory use
        batch_images = [self.images[self.indices[i]] for i in range(batch_start,batch_end)]
        batch_masks = [self.labels[self.indices[i]] for i in range(batch_start,batch_end)]

        return batch_images, batch_masks
        
    def _loadBatchFromDisk(self, batch_start, batch_end):
        batch_images = []
        batch_masks = []
        for i in range(batch_start, batch_end):
            train_img, train_mask = self.parent.data.createImageLabelPair(self.indices[i],validation=self.validation)
            batch_images.append(train_img)
            batch_masks.append(train_mask)
        return batch_images, batch_masks
        
        
    def on_epoch_end(self):
        if not self.validation:
            random.shuffle(self.indices)
        

class PredictionDataGenerator(Sequence):
    def __init__(self, parent, image):
        self.parent = parent
        self.lock = threading.Lock()
        self.inMemory  = parent.TrainInMemory

        channels = self.parent.Channels
        self.image = image.reshape(image.shape[0], image.shape[1], channels)
        self.t_width = self.parent.augmentation.outputwidth - self.parent.ModelOverlap
        self.t_height = self.parent.augmentation.outputheight - self.parent.ModelOverlap 
        
        
        if self.image.shape[1] < self.parent.augmentation.outputwidth or self.image.shape[0] < self.parent.augmentation.outputheight:
            w = max(0,self.parent.augmentation.outputwidth - self.image.shape[1])
            h = max(0,self.parent.augmentation.outputheight - self.image.shape[0])
            #if self.image.shape
            pad = ((0,h), (0,w), (0, 0))
            self.image = np.pad(self.image, pad, 'reflect')

        pad = ((self.parent.ModelOverlap//2, self.parent.ModelOverlap//2), (self.parent.ModelOverlap//2, self.parent.ModelOverlap//2), (0, 0))       
        self.image = np.pad(self.image, pad, 'reflect')
           
        self.num_in_width = np.ceil(image.shape[1] / self.t_width)
        self.num_in_height = np.ceil(image.shape[0] / self.t_height)
        self.numTiles = int(self.num_in_width * self.num_in_height)

    def __len__(self):
        return int(np.ceil(self.num_in_width * self.num_in_height / (self.parent.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            batch_start = idx * self.parent.batch_size
            batch_end = min(batch_start + self.parent.batch_size, self.numTiles)


            batch = []
            for currentTile in range(batch_start,batch_end):      
                w = (currentTile % self.num_in_width) * self.t_width
                h = (currentTile // self.num_in_width) * self.t_height

                w_0 = max(w, 0)
                h_0 = max(h, 0)
                
                # split factor need to be reworked and reasonable, get it from network
                h_til = min(h_0+self.t_height+self.parent.ModelOverlap, self.image.shape[0])
                w_til = min(w_0+self.t_width+self.parent.ModelOverlap,  self.image.shape[1])
                
                # a full tile at the borders results in better segmentations
                if h_til == self.image.shape[0]:
                    h_0 = max(self.image.shape[0] - (self.t_height + self.parent.ModelOverlap), 0)
                if w_til == self.image.shape[1]:
                    w_0 = max(self.image.shape[1] - (self.t_width + self.parent.ModelOverlap), 0)

                tile = self.image[int(h_0):int(h_til), int(w_0):int(w_til)]
                batch.append(tile)
                
            img = self.parent.data.preprocessImage(np.asarray(batch))
            return self.parent.augmentation.checkModelSizeCompatibility(img)
