# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:54:36 2019

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import tensorflow as tf



import cv2
import glob
import os
import numpy as np
from enum import Enum


import dl.dl_data as dl_data
import dl.dl_models as dl_models
import dl.dl_losses as dl_losses

import utils.Contour


class dlMode(Enum):
    classification = 1
    segmentation = 2
    # add more: tracking, 3Dsegmentation



class DeepLearning():
    def __init__(self):
        
        self.TrainInMemory = True
        self.ImageScaleFactor = 0.25
        self.initialized = False
        
        self.ModelType = 0
        self.Model = None
        self.Models = dl_models.Models()
        self.Mode = dlMode.segmentation
        self.NumClasses = 2
        
        # Training settings
        self.MonoChrome = True
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 1e-5
        
        
        
    def initModel(self, numClasses = 2):
        self.NumClasses = numClasses
        try:
            self.Model = self.Models.getModel(self.Mode, self.ModelType, self.NumClasses, monochrome = self.MonoChrome)
        except:
            return False
        if self.Model:
            self.initialized = True
            return True
       
    def Train(self, trainingimages_path, traininglabels_path):
        if not self.initialized or trainingimages_path is None or traininglabels_path is None:
            return False
        
        if self.TrainInMemory:
            x,y = dl_data.loadTrainingDataSet(trainingimages_path, traininglabels_path, self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
            train_generator = dl_data.TrainingDataGenerator_inMemory(x,y, batch_size = self.batch_size, numclasses = self.NumClasses)
        else:
            x,y = dl_data.getTrainingDataset(trainingimages_path, traininglabels_path)
            train_generator = dl_data.TrainingDataGenerator_fromDisk(x,y, batch_size = self.batch_size, numclasses = self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
        
        adam = Adam(lr=self.learning_rate)
        if self.NumClasses > 2:
            loss = dl_losses.focal_loss
        else:
#            loss = 'binary_crossentropy'
            loss = dl_losses.focal_loss_binary
        # to do implement binary iou
        self.Model.compile(optimizer=adam, loss=loss, metrics=[MeanIoU(num_classes=self.NumClasses)])   
        
        ## this needs more investigation for some reasons, fit_generator is much slower than fit
        try:
            self.Model.fit_generator(train_generator, steps_per_epoch=len(x)/self.batch_size,verbose=1, callbacks=None, epochs=self.epochs, workers = 20)
        except:
            return False
        return True
       
            
    def PredictImage(self, image):    
        if not self.initialized or image is None:
            return None

        ### to do -> ensure matching size, either here or in the model
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))
        split_factor = 16
        pad = (split_factor - image.shape[0] % split_factor, split_factor - image.shape[1] % split_factor)
        pad = (pad[0] % split_factor, pad[1] % split_factor)            
        if pad != (0,0):
            image = np.pad(image, ((0, pad[0]), (0, pad[1])),'constant', constant_values=(0, 0))
                       
        if not self.MonoChrome and len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
        
        if len(image.shape) == 2 :
            image = image[np.newaxis, :, :, np.newaxis].astype('float')/255.
        elif len(image.shape) == 3:
            image = image[np.newaxis, :, :, :].astype('float')/255.
        else:
            raise ('wrong image format')

        # convert_to_tensor is not necessary but improves performance because it avoids retracing
        pred = self.Model.predict(tf.convert_to_tensor(image,np.float32))
        if self.NumClasses > 2:
            pred = np.squeeze(np.argmax(pred, axis = 3))
        else:
            pred = np.squeeze(pred)
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
        pred = pred[0:pred.shape[0]-pad[0],0:pred.shape[1]-pad[1]]
        pred = pred.astype('uint8')
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return pred

        
    
    def setMode(self, mode):
        self.Mode = mode
            
    def Reset(self):
        self.initialized = False
    
    def LoadModel(self, modelpath):
        self.Model = load_model (modelpath)
        if self.Model:
            self.initialized = True
    
    def SaveModel(self, modelpath):
        if self.initialized:
            self.Model.save(modelpath)
    
    