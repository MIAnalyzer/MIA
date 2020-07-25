# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:54:36 2019

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, SGD
from tensorflow.keras import losses
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import tensorflow as tf


import math
from inspect import signature
import cv2
import glob
import os
import numpy as np
from enum import Enum
import threading
import traceback
import sys

import dl.dl_datagenerator as dl_datagenerator
# import dl.dl_models as dl_models
import dl.dl_losses as dl_losses
import dl.dl_metrics as dl_metrics
import dl.dl_training_record as dl_training_record
import dl.dl_hed as dl_hed
import dl.dl_imagedata as dl_imagedata
import dl.dl_augment as dl_augment
import dl.dl_lrschedule as dl_schedule
from dl.dl_segmentation import Segmentation
from dl.dl_objectcounting import ObjectCounting
from dl.dl_classification import Classification
from dl.dl_undefinedmode import UndefinedMode
from dl.dl_mode import dlMode

from utils.Observer import dlObservable

import utils.Contour
import multiprocessing


class dlLoss(Enum):
    focal = 1
    # add optional class weights

class dlMetric(Enum):
    iou = 1

class dlOptim(Enum):
    Adam = 1
    AdaMax = 2
    AdaDelta = 3
    AdaGrad = 4
    SGDNesterov = 5
    

class DeepLearning(dlObservable):
    def __init__(self):
        super().__init__()
        self.TrainInMemory = True
        self.useWeightedDistanceMap = False
        self.ImageScaleFactor = 0.5
        self.worker = multiprocessing.cpu_count() // 2
        self.hed = dl_hed.HED_Segmentation()
        self.data = dl_imagedata.ImageData(self)
        self.augmentation = dl_augment.ImageAugment()
        self.record = dl_training_record.TrainingRecording(self)
        
        self.ModelType = 0
        self.Model = None
        # split factor need to be reworked and reasonable, get it from network
        self.split_factor = 48

        
        # self.Models = dl_models.Models()
        self.Mode = Segmentation(self)
        
        self.observer = None
        
        # Training settings
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 0.001
        self.lrschedule = dl_schedule.LearningRateSchedule(self)


        self.Loss = dlLoss.focal
        self.Metric = dlMetric.iou
        self.Optimizer = dlOptim.Adam

    @property
    def NumClasses(self):
        if not self.initialized:
            raise ('model not initialized')
        return self.Model.output_shape[3]
    
    @property
    def MonoChrome(self):        
        if not self.initialized:
            raise ('model not initialized')
        return True if self.Model.input_shape[3] == 1 else False

    @property
    def initialized(self):
        return True if self.Model else False
     
    @property
    def WorkingMode(self):
        return self.Mode.type
    
    @WorkingMode.setter
    def WorkingMode(self, mode):
        if mode == dlMode.Segmentation:
            self.Mode = Segmentation(self)
        elif mode == dlMode.Object_Counting:
            self.Mode = ObjectCounting(self)    
        elif mode == dlMode.Classification:
            self.Mode = Classification(self)
        else:
            self.Mode = UndefinedMode(self)

    def interrupt(self):
        self.record.interrupt = True
          
    def initModel(self, numClasses, MonoChrome):
        try:
            # self.Model = self.Models.getModel(self.Mode, self.ModelType, numClasses, monochrome = MonoChrome)
            self.Model = self.Mode.getModel(numClasses, MonoChrome)
            self.record.reset()
            return self.initialized
        except:
            self.notifyError()
            return False

       
    def Train(self, trainingimages_path, traininglabels_path):
        if not self.initialized or trainingimages_path is None or traininglabels_path is None:
            return False

        thread = threading.Thread(target=self.executeTraining, args=(trainingimages_path, traininglabels_path))
        thread.daemon = True
        thread.start()

    
    def executeTraining(self, trainingimages_path, traininglabels_path):
        try:
            self.data.initTrainingDataset(trainingimages_path, traininglabels_path)
            if not self.data.initialized:
                return

            train_generator = dl_datagenerator.TrainingDataGenerator(self)
            if self.data.validationData():
                val_generator = dl_datagenerator.TrainingDataGenerator(self, validation = True)
            else: 
                val_generator = None    
            self.Model.compile(optimizer=self._getOptimizer(), loss=self._getLoss(), metrics=self._getMetrics()) 
            self.Model.fit(train_generator,validation_data=val_generator, verbose=1, callbacks=self._getTrainingCallbacks(), epochs=self.epochs, workers = self.worker)
        except:
            self.notifyTrainingFinished()
            self.notifyError()
 
    def PredictImage_async(self, image):    
        if not self.initialized or image is None:
            return

        thread = threading.Thread(target=self.executePrediction, args=(image,))
        thread.daemon = True
        thread.start()   
        
    def executePrediction(self, image):
        # prediction would be much faster when all images where predicted from here compared to 1-by-1
        try:
            prediction = self.Mode.PredictImage(image)
        except:
            self.notifyError() 
        self.notifyPredictionFinished(prediction)


    def AutoSegment(self, image):    
        if image is None:
            return None

        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))
        
        pred = self.hed.applyHED(image)
        _,thresh = cv2.threshold(pred,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return ret

            
    def parameterFit(self, nclasses, mono):
        if not self.initialized:
            return False    
        if nclasses == 2:
            nclasses = 1
        if nclasses == self.NumClasses and mono == self.MonoChrome:
            return True       
        else:
            return False
            
            
    def Reset(self):
        self.Model = None
    
    def LoadModel(self, modelpath):
        self.Model = load_model (modelpath)
    
    def SaveModel(self, modelpath):
        if self.initialized:
            self.Model.save(modelpath)
            
    def _getTrainingCallbacks(self):
        callbacks = [self.record]
        lr = self.lrschedule.LearningRateCallBack()
        if lr is not None:
            callbacks.append(lr)
        return callbacks
    
    def _getLoss(self):
        if self.Loss == dlLoss.focal:
            return dl_losses.focal_loss_function(binary = self.NumClasses <= 2, weighted = self.useWeightedDistanceMap)

    def _getMetrics(self):
        if self.Metric == dlMetric.iou:
            return dl_metrics.iou_function(binary = self.NumClasses <= 2, weighted = self.useWeightedDistanceMap)

    def _getOptimizer(self):
        if self.Optimizer == dlOptim.Adam:
            optim = Adam(learning_rate=self.learning_rate)
        if self.Optimizer == dlOptim.AdaMax:
            optim = AdaMax(learning_rate=self.learning_rate)
        if self.Optimizer == dlOptim.AdaDelta:
            optim = AdaDelta(learning_rate=self.learning_rate)
        if self.Optimizer == dlOptim.AdaGrad:
            optim = Adagrad(learning_rate=self.learning_rate)
        if self.Optimizer == dlOptim.SGDNesterov:
            optim = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True)  
        return optim
    
    def setOptimizer(self, opti):
        self.Optimizer = opti
        if self.Optimizer == dlOptim.Adam:
            fun = Adam
        if self.Optimizer == dlOptim.AdaMax:
            fun = Adamax
        if self.Optimizer == dlOptim.AdaDelta:
            fun = Adadelta
        if self.Optimizer == dlOptim.AdaGrad:
            fun = Adagrad
        if self.Optimizer == dlOptim.SGDNesterov:
            fun = SGD  
        self.learning_rate = signature(fun).parameters['learning_rate'].default