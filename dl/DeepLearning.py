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

import math

import cv2
import glob
import os
import numpy as np
from enum import Enum


import dl.dl_data as dl_data
import dl.dl_models as dl_models
import dl.dl_losses as dl_losses
import dl.dl_metrics as dl_metrics
import dl.dl_training_record as dl_training_record
import dl.dl_hed as dl_hed

import utils.Contour
import multiprocessing

class dlMode(Enum):
    classification = 1
    segmentation = 2
    # add more: tracking, 3Dsegmentation

class dlLoss(Enum):
    focal = 1

class dlMetric(Enum):
    iou = 1

class dlOptim(Enum):
    adam = 1

class DeepLearning():
    def __init__(self):
        
        self.TrainInMemory = True
        self.useWeightedDistanceMap = False
        self.ImageScaleFactor = 0.3
        self.worker = multiprocessing.cpu_count() // 2
        self.hed = dl_hed.HED_Segmentation()
        
        self.ModelType = 0
        self.Model = None
        self.Models = dl_models.Models()
        self.Mode = dlMode.segmentation
        
        # Training settings
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 1e-5

        self.Loss = dlLoss.focal
        self.Metric = dlMetric.iou
        self.Optimizer = dlOptim.adam
        
        
    def initModel(self, numClasses, MonoChrome):
        try:
            self.Model = self.Models.getModel(self.Mode, self.ModelType, numClasses, monochrome = MonoChrome)
            return self.initialized()
        except:
            return False
       
    def Train(self, trainingimages_path, traininglabels_path):
        if not self.initialized() or trainingimages_path is None or traininglabels_path is None:
            return False

        if self.TrainInMemory:
            x,y = dl_data.loadTrainingDataSet(trainingimages_path, traininglabels_path, self.useWeightedDistanceMap, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome())
            train_generator = dl_data.TrainingDataGenerator_inMemory(self,x,y)
        else:
            x,y = dl_data.getTrainingDataset(trainingimages_path, traininglabels_path)
            train_generator = dl_data.TrainingDataGenerator_fromDisk(self,x,y)

        self.Model.compile(optimizer=self.__getOptimizer(), loss=self.__getLoss(), metrics=self.__getMetrics())  

        ## this needs more investigation for some reasons, fit_generator is much slower than fit
        callbacks = [dl_training_record.TrainingRecording(self)]
        try:
            self.Model.fit(train_generator,verbose=1, callbacks=callbacks, epochs=self.epochs, workers = self.worker)
        except:
            return False
        return True
       
            
    def PredictImage(self, image):    
        if not self.initialized() or image is None:
            return None

        ### to do -> ensure matching size, either here or in the model
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))

        if len(image.shape) == 2:
            if not self.MonoChrome():
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
            else:
                image = image[..., np.newaxis]
        elif len(image.shape) == 3:
            if self.MonoChrome() and image.shape[2] > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
                # atm I dont know if bgr2gray squeezes
                if len(image.shape) == 2:
                    image = image[..., np.newaxis]
        else:
            raise ('wrong image format')

        split_factor = 16
        pad = (split_factor - image.shape[0] % split_factor, split_factor - image.shape[1] % split_factor)
        pad = (pad[0] % split_factor, pad[1] % split_factor)            
        if pad != (0,0):
            image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0,0)),'constant', constant_values=(0, 0))

        image = image[np.newaxis, ...].astype('float')/255.
        # convert_to_tensor is not necessary but improves performance because it avoids retracing
        pred = self.Model.predict(tf.convert_to_tensor(image,np.float32))
        if self.NumClasses() > 2:
            pred = np.squeeze(np.argmax(pred, axis = 3))
        else:
            pred = np.squeeze(pred)
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
        pred = pred[0:pred.shape[0]-pad[0],0:pred.shape[1]-pad[1]]
        pred = pred.astype('uint8')
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return pred

    def AutoSegment(self, image):    
        if image is None:
            return None

        pred = self.hed.applyHED(image)
        _,thresh = cv2.threshold(pred,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def NumClasses(self):
        if not self.initialized():
            raise ('model not initialized')
        return self.Model.output_shape[3]
    
    def MonoChrome(self):        
        if not self.initialized():
            raise ('model not initialized')
        return True if self.Model.input_shape[3] == 1 else False

    def parameterFit(self, nclasses, mono):
        if not self.initialized():
            return False    
        if nclasses == 2:
            nclasses = 1
        if nclasses == self.NumClasses() and mono == self.MonoChrome():
            return True       
        else:
            return False

    def initialized(self):
        return True if self.Model else False

    def setMode(self, mode):
        self.Mode = mode
            
    def Reset(self):
        self.Model = None
    
    def LoadModel(self, modelpath):
        self.Model = load_model (modelpath)
    
    def SaveModel(self, modelpath):
        if self.initialized():
            self.Model.save(modelpath)
    
    def __getLoss(self):
        if self.Loss == dlLoss.focal:
            return dl_losses.focal_loss_function(binary = self.NumClasses() <= 2, weighted = self.useWeightedDistanceMap)

    def __getMetrics(self):
        if self.Metric == dlMetric.iou:
            return dl_metrics.iou_function(binary = self.NumClasses() <= 2, weighted = self.useWeightedDistanceMap)

    def __getOptimizer(self):
        if self.Optimizer == dlOptim.adam:
            optim = Adam(lr=self.learning_rate)
        return optim