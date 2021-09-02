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
import threading
import traceback
import sys

import dl.training.datagenerator as datagenerator
import dl.training.training_record as training_record
import dl.training.augment as augment
import dl.training.lrschedule as schedule
import dl.machine_learning.hed as hed
import dl.machine_learning.dextr.dextr_segmentation as dextr
import dl.machine_learning.grabcut_segmentation as gcs
import dl.data.imagedata as imagedata
from dl.method.segmentation import Segmentation
from dl.method.objectcounting import ObjectCounting
from dl.method.classification import Classification
from dl.method.undefinedmode import UndefinedMode
from dl.method.mode import dlMode, dlPreprocess
from dl.optimizer.optimizer import Optimizer

from utils.Observer import dlObservable
import gc

import utils.shapes.Contour
import multiprocessing

class DeepLearning(dlObservable):
    def __init__(self):
        super().__init__()
        self.TrainInMemory = True
        self.ImageScaleFactor = 0.5
        self.worker = multiprocessing.cpu_count() // 2
        self.hed = hed.HED_Segmentation()
        self.dextr = dextr.DEXTR_Segmentation()
        self.grabcut = gcs.GrabCutSegmentation()
        self.data = imagedata.ImageData(self)
        self.augmentation = augment.ImageAugment(self)
        self.record = training_record.TrainingRecording(self)
        
        self.Model = None
        # split factor and border need to be reworked and reasonable, get it from network
        self.split_factor = 32
        self.borderremoval = 112

        self.interrupted = False

        self.train_generator = None
        self.val_generator = None

        self.resumeEpoch = 0

        self.Mode = Segmentation(self)
        
        self.observer = None
        
        self.tta = True
        
        # Training settings
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 0.001
        self.preprocess = dlPreprocess.scale
        self.lrschedule = schedule.LearningRateSchedule(self)
        self.optimizer = Optimizer(self)
        
        # segmentation settings
        self.seg_useWeightedDistanceMap = False
        
        # object detection settings
        self.od_kernel_stdev = 3
        self.od_peak_val = 125
        

    @property
    def NumClasses(self):
        # returns model output
        if not self.initialized:
            raise ('model not initialized')
        return self.Model.output_shape[-1]

    @property
    def NumClasses_real(self):
        # returns real number of classes to differentiate
        classes = self.NumClasses
        if classes == 1:
            classes = 2
        return classes
    
    @property
    def MonoChrome(self):        
        if not self.initialized:
            raise ('model not initialized')
        return True if self.Model.input_shape[3] == 1 else False

    @property
    def InputSize(self):        
        if not self.initialized:
            raise ('model not initialized')
        return self.Model.input_shape[1:3]

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
        # we reset here to avoid using a wrong model
        self.Model = None
     
    @property
    def LabelFolderName(self):
        return self.Mode.type.name + '_labels'

    def interrupt(self):
        self.record.interrupt = True
        self.interrupted = True
        self.resumeEpoch = self.record.currentepoch
          
    def initModel(self, numClasses, MonoChrome):
        try:
            if numClasses == 2:
                numClasses = 1
            self.Model = self.Mode.getModel(numClasses, 1 if MonoChrome is True else 3)
            return self.initialized
        except:
            self.notifyError()
            return False

    def cleanMemory(self):
        # where/when do we need to put this
        gc.collect()
        tf.keras.backend.clear_session()
       
    def startTraining(self, trainingimages_path, validation_path):
        if not self.initialized or trainingimages_path is None:
            return False
        self.cleanMemory()

        thread = threading.Thread(target=self.initData_StartTraining, args=(trainingimages_path,validation_path))
        thread.daemon = True
        thread.start()

    def resumeTraining(self):
        if not self.initialized:
            return False
        self.cleanMemory()
        thread = threading.Thread(target=self.Training)
        thread.daemon = True
        thread.start()

    
    def initData_StartTraining(self, trainingimages_path, validation_path):
        try:
            self.record.reset()
            self.interrupted = False
            self.data.initTrainingDataset(trainingimages_path, validation_path)
            if not self.data.initialized() or self.interrupted:
                self.notifyTrainingFinished()
                return
            self.train_generator = datagenerator.TrainingDataGenerator(self)
            if self.data.validationData():
                self.val_generator = datagenerator.TrainingDataGenerator(self, validation = True)
            else: 
                self.val_generator = None 
            self.Model.compile(optimizer=self.optimizer.getOptimizer(), loss=self.Mode.loss.getLoss(), metrics=self.Mode.metric.getMetric()) 
            self.resumeEpoch = 0

            self.Training()
        except:
            self.notifyTrainingFinished()
            self.notifyError()

    def Training(self):
        try:
            if not self.train_generator:
                self.notifyTrainingFinished()
                return
            self.Model.fit(self.train_generator,validation_data=self.val_generator, initial_epoch = self.resumeEpoch, verbose=1, callbacks=self._getTrainingCallbacks(), epochs=self.epochs, workers = self.worker)
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

    def AssistedSegmentation(self, image, extremepoints):      
        # no resizing here, as image is cropped to target network size (512,512) anyway
        # if resize is necessary remember to resize points, too
        return self.dextr.predict(image, extremepoints)
    
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
        self.Model = self.Mode.loadmodel(modelpath)
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
    
