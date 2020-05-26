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
# import dl.dl_augment as dl_augment
import dl.dl_utils as dl_utils
from utils.Contour import LoadLabel



class TrainingDataGenerator_inMemory(Sequence):
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = parent.batch_size
        self.lock = threading.Lock() 
        images,labels = parent.data.loadTrainingDataSet()
        self.parent.augmentation.initAugmentation()
        self.images = images
        self.labels = labels
        self.indices = list(range(0,images.shape[0]))
        self.numImages = len(self.indices)
        random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.numImages / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.numImages)

            batch_images = self.images[self.indices[batch_start:batch_end]]
            batch_masks = self.labels[self.indices[batch_start:batch_end]]

            img, mask = self.parent.augmentation.augment(batch_images,batch_masks)
            # img, mask = dl_augment.augment(batch_images,batch_masks)         
            img = self.parent.data.preprocessImage(img)     
            mask = self.parent.data.preprocessLabel(mask)
            return img, mask
        
    def on_epoch_end(self):
        random.shuffle(self.indices)


class TrainingDataGenerator_fromDisk(Sequence):
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = parent.batch_size
        self.lock = threading.Lock()
        self.images = self.parent.data.ImagesPaths
        self.labels = self.parent.data.LabelsPaths
        self.parent.augmentation.initAugmentation()
        self.channels = 1 if parent.MonoChrome() is True else 3
        self.numClasses = parent.NumClasses()
        self.scalefactor = parent.ImageScaleFactor
        self.indices = list(range(0,len(self.images)))
        self.numImages = len(self.images)
        random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.numImages / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            batch_images = []
            batch_masks = []
            batch_start = idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.numImages)
            for i in range(batch_start, batch_end):
                
                train_img, train_mask = self.parent.data.createImageLabelPair(i)

                batch_images.append(train_img)
                batch_masks.append(train_mask)

            img, mask = self.parent.augmentation.augment(batch_images,batch_masks)
            # img, mask = dl_augment.augment(batch_images, batch_masks)

            img = self.parent.data.preprocessImage(img)     
            mask = self.parent.data.preprocessLabel(mask)
            return img, mask
        
    def on_epoch_end(self):
        random.shuffle(self.indices)
    
    

