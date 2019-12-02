# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:47:31 2019

@author: Koerber
"""


import os
import cv2
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import random
import threading
import dl_augment
from Contour import LoadLabel

from tensorflow.python.keras.utils.data_utils import Sequence


class TrainingDataGenerator_inMemory(Sequence):
    def __init__(self, images, labels, batch_size,numclasses):
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.numClasses = numclasses
        
        self.indices = list(range(0,images.shape[0]))
        self.numImages = len(self.indices)
        random.shuffle(self.indices)
        self.cnt = 0

    
    def __len__(self):
        return int(np.ceil(self.numImages / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            batch_images = self.images[self.indices[self.cnt:self.cnt+self.batch_size]]
            batch_masks = self.labels[self.indices[self.cnt:self.cnt+self.batch_size]]
            self.cnt += self.batch_size
            
            img, mask = dl_augment.augment(batch_images,batch_masks)             
            img = img.astype('float')/255.
            if self.numClasses > 2:
                mask = to_categorical(mask, num_classes=self.numClasses)
            else:
                mask = mask.astype('bool')

            return img, mask
        
    def on_epoch_end(self):
        self.cnt=0
        random.shuffle(self.indices)


class TrainingDataGenerator_fromDisk(Sequence):
    def __init__(self, images_path, labels_path, batch_size, numclasses, scalefactor=1, monochrome = False):
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.images = images_path
        self.labels = labels_path
        self.batch_size = batch_size
        self.channels = 1 if monochrome is True else 3
        self.numClasses = numclasses
        self.scalefactor = scalefactor
        
        
        self.indices = list(range(0,len(self.images)))
        self.numImages = len(self.images)
        random.shuffle(self.indices)
        self.cnt = 0

    
    def __len__(self):
        return int(np.ceil(self.numImages / float(self.batch_size)))

    def __getitem__(self, idx):
        with self.lock:
            batch_images = []
            batch_masks = []
            for i in range(self.cnt, self.cnt+self.batch_size):
                if i >= self.numImages:
                    break
                if self.channels == 1:
                    train_img = cv2.imread(self.images[self.indices[i]], cv2.IMREAD_GRAYSCALE).astype('uint8')    
                else:
                    train_img = cv2.imread(self.images[self.indices[i]], cv2.IMREAD_COLOR).astype('uint8')
                
                #train_mask = cv2.imread(self.labels[self.indices[i]], cv2.IMREAD_GRAYSCALE).astype('uint8')
                train_mask = LoadLabel(self.labels[self.indices[i]], train_img.shape[0], train_img.shape[1])
                
                    
                width = int(train_img.shape[1]*self.scalefactor)
                height= int(train_img.shape[0]*self.scalefactor)
                
                train_img =  cv2.resize(train_img, (width, height))
                train_mask = cv2.resize(train_mask, (width, height), interpolation = cv2.INTER_NEAREST )
                
                # remove unlabelled data
                _, thresh = cv2.threshold(train_mask, 0, 255,  cv2.THRESH_BINARY)
                train_img = cv2.bitwise_and(train_img,train_img, mask=thresh)
                
                # remove background
                train_mask[np.where(train_mask == 255)] = 0
                
                train_mask = train_mask.reshape(height, width, 1)
                train_img = train_img.reshape(height, width, self.channels)

                
                batch_images.append(train_img)
                batch_masks.append(train_mask)

            
            img, mask = dl_augment.augment(batch_images, batch_masks)
            self.cnt+=self.batch_size
            
            img = img.astype('float')/255.
            if self.numclasses > 2:
                mask = to_categorical(mask, num_classes=self.numClasses)
            else:
                mask = mask.astype('bool')

            return img, mask
        
    def on_epoch_end(self):
        self.cnt=0
        random.shuffle(self.indices)
    
    
    
def loadTrainingDataSet(imagepath, labelpath, numclasses, scalefactor=1, monochrome = False):
   
    images, labels = getTrainingDataset(imagepath, labelpath)
   
    test = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)
    width = int(test.shape[1]*scalefactor)
    height = int(test.shape[0]*scalefactor)
    
    
    channels = 1 if monochrome is True else 3
    
    # now: all images are resized to the size of the first image 
    # Alternatively: all images could be padded to the largest image size
    img = np.zeros((len(images), height, width, channels)).astype('uint8')
    mask = np.zeros((len(images), height, width, 1)).astype('uint8')
    
    
    for i in range (len(images)):
        if monochrome:
            train_img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)      
        else:
            train_img = cv2.imread(images[i], cv2.IMREAD_COLOR)
            
        #train_mask = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        train_mask = LoadLabel(labels[i], train_img.shape[0], train_img.shape[1])
         
        train_img =  cv2.resize(train_img, (width, height))
        train_mask = cv2.resize(train_mask, (width, height), interpolation = cv2.INTER_NEAREST )

        # remove unlabelled data
        _, thresh = cv2.threshold(train_mask, 0, 255,  cv2.THRESH_BINARY)
        train_img = cv2.bitwise_and(train_img,train_img, mask=thresh)
        
        # remove background
        train_mask[np.where(train_mask == 255)] = 0

        # concat
        train_mask = train_mask.reshape(height, width, 1)
        train_img = train_img.reshape(height, width, channels)
        img[i] = train_img
        mask[i] = train_mask


        
    return img, mask



def getTrainingDataset(imagepath, labelpath):
    
    images = glob.glob(os.path.join(imagepath,'*.*'))
    labels = glob.glob(os.path.join(labelpath,'*.*'))
      
    labels = [x for x in labels if (x.endswith(".tif") or x.endswith(".npy")) or x.endswith(".npz")]
    
    image_names = [os.path.splitext(os.path.basename(each))[0] for each in images]
    label_names = [os.path.splitext(os.path.basename(each))[0] for each in labels]
    
    intersection = list(set(image_names).intersection(label_names))
    if intersection == list():
        return None, None
    
    images = [each for each in images if os.path.splitext(os.path.basename(each))[0] in intersection]
    labels = [each for each in labels if os.path.splitext(os.path.basename(each))[0] in intersection]
    images.sort()
    labels.sort()
    
    return images, labels






