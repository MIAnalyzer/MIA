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

from tensorflow.python.keras.utils.data_utils import Sequence
import dl.dl_utils as dl_utils

class TrainingDataGenerator(Sequence):
    def __init__(self, parent, validation = False):
        self.parent = parent
        self.batch_size = parent.batch_size
        self.lock = threading.Lock()
        self.inMemory  = parent.TrainInMemory
        self.validation = validation
        if self.inMemory:
            self.images,self.labels = parent.data.loadTrainingDataSet(validation = self.validation)
        else:
            self.images = self.parent.data.getImagePaths(self.validation)
            self.labels = self.parent.data.getLabelPaths(self.validation)
            
        self.parent.augmentation.initAugmentation()
        # self.indices = list(range(0,len(self.images)))
        self.indices = self.parent.data.getTileIndices(validation, equalTilesperImage = not self.inMemory)
        self.numImages = len(self.images)
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
        with self.lock:

            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.numTiles)
            if self.inMemory:
                batch_images,batch_masks = self._loadBatchFromMemory(batch_start, batch_end)
            else:
                batch_images,batch_masks = self._loadBatchFromDisk(batch_start, batch_end)

    
            img, mask = self.parent.augmentation.augment(batch_images,batch_masks, self.validation)
            img = self.parent.data.preprocessImage(img)     
            mask = self.parent.data.preprocessLabel(mask)

            return img, mask
        
    def _loadBatchFromMemory(self, batch_start, batch_end): 
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
        

    

