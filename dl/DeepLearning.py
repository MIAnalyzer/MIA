# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:54:36 2019

@author: Koerber
"""


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
import threading
import traceback
import sys

import dl.training.datagenerator as datagenerator
import dl.training.training_record as training_record
import dl.training.augment as augment
import dl.training.lrschedule as schedule
import dl.models.hed as hed
import dl.machine_learning.grabcut_segmentation as gc
import dl.data.imagedata as imagedata
from dl.method.segmentation import Segmentation
from dl.method.objectcounting import ObjectCounting
from dl.method.classification import Classification
from dl.method.undefinedmode import UndefinedMode
from dl.method.mode import dlMode
from dl.optimizer.optimizer import Optimizer
from dl.postprocessing.tracking import ObjectTracking

from utils.Observer import dlObservable


import utils.shapes.Contour
import multiprocessing

class DeepLearning(dlObservable):
    def __init__(self):
        super().__init__()
        self.TrainInMemory = True
        self.useWeightedDistanceMap = False
        self.ImageScaleFactor = 0.5
        self.worker = multiprocessing.cpu_count() // 2
        self.hed = hed.HED_Segmentation()
        self.grabcut = gc.GrabCutSegmentation()
        self.data = imagedata.ImageData(self)
        self.augmentation = augment.ImageAugment()
        self.record = training_record.TrainingRecording(self)
        
        self.ModelType = 0
        self.Model = None
        # split factor need to be reworked and reasonable, get it from network
        self.split_factor = 48
        
        self.tracking = ObjectTracking(self)
        self.interrupted = False

    
        self.Mode = Segmentation(self)
        
        self.observer = None
        
        # Training settings
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 0.001
        self.lrschedule = schedule.LearningRateSchedule(self)


        self.optimizer = Optimizer(self)

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
        self.interrupted = True
          
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
            self.interrupted = False
            self.data.initTrainingDataset(trainingimages_path, traininglabels_path)
            if not self.data.initialized() or self.interrupted:
                self.notifyTrainingFinished()
                return

            train_generator = datagenerator.TrainingDataGenerator(self)
            if self.data.validationData():
                val_generator = datagenerator.TrainingDataGenerator(self, validation = True)
            else: 
                val_generator = None    
            self.Model.compile(optimizer=self.optimizer.getOptimizer(), loss=self.Mode.loss.getLoss(), metrics=self.Mode.metric.getMetric()) 
            if self.interrupted:
                self.notifyTrainingFinished()
                return
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
    
    def SemiAutoSegment(self,image, mask = None, train=True):
        if image is None:
            return None

        width = image.shape[1]
        height = image.shape[0]
        
        red_height = int(width*self.ImageScaleFactor)
        red_width = int(height*self.ImageScaleFactor)
        
        while red_height*red_width > 500000:
        # with more than 500k pixels grabcut becomes disproportional slow
        # to use on higher res images scale down train and infer tiles
            red_height *= 0.9
            red_width *= 0.9

        image = cv2.resize(image, (int(red_width), int(red_height)))
        mask = cv2.resize(mask, (int(red_width), int(red_height)),interpolation=cv2.INTER_NEAREST)
        
        if train:
             pred = self.grabcut.trainAndSegment(image, mask)
        else:
            pred = self.grabcut.Segment(image)
        ret = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return ret
        
       
    def calculateTracking(self, labelsequence):
        self.tracking.performTracking(labelsequence)
       
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
        self.Model = load_model(modelpath)
        self.record.reset()
    
    def SaveModel(self, modelpath):
        if self.initialized:
            self.Model.save(modelpath)
            
    def _getTrainingCallbacks(self):
        callbacks = [self.record]
        lr = self.lrschedule.LearningRateCallBack()
        if lr is not None:
            callbacks.append(lr)
        return callbacks
    

