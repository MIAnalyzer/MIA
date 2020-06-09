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


import dl.dl_datagenerator as dl_datagenerator
import dl.dl_models as dl_models
import dl.dl_losses as dl_losses
import dl.dl_metrics as dl_metrics
import dl.dl_training_record as dl_training_record
import dl.dl_hed as dl_hed
import dl.dl_imagedata as dl_imagedata
import dl.dl_augment as dl_augment
import dl.dl_lrschedule as dl_schedule

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
    Adam = 1
    AdaMax = 2
    AdaDelta = 3
    AdaGrad = 4
    SGDNesterov = 5
    

class DeepLearning():
    def __init__(self):
        
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
        self.split_factor = 32
        self.PredictFullImage = True
        
        self.Models = dl_models.Models()
        self.Mode = dlMode.segmentation
        
        # Training settings
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 0.001
        self.lrschedule = dl_schedule.LearningRateSchedule(self)

        self.Loss = dlLoss.focal
        self.Metric = dlMetric.iou
        self.Optimizer = dlOptim.Adam
        
        
    def initModel(self, numClasses, MonoChrome):
        # try:
        if True:
            self.Model = self.Models.getModel(self.Mode, self.ModelType, numClasses, monochrome = MonoChrome)
            return self.initialized()
        # except:
        #     return False
       
    def Train(self, trainingimages_path, traininglabels_path):
        if not self.initialized() or trainingimages_path is None or traininglabels_path is None:
            return False

        self.data.initTrainingDataset(trainingimages_path, traininglabels_path)
        if self.TrainInMemory:
            train_generator = dl_datagenerator.TrainingDataGenerator_inMemory(self)
        else:
            train_generator = dl_datagenerator.TrainingDataGenerator_fromDisk(self)

        self.Model.compile(optimizer=self._getOptimizer(), loss=self._getLoss(), metrics=self._getMetrics())  


        callbacks = [self.record]
        lr = self.lrschedule.LearningRateCallBack()
        if lr is not None:
            callbacks.append(lr)
        #try:
        if True:
            self.Model.fit(train_generator,verbose=1, callbacks=callbacks, epochs=self.epochs, workers = self.worker)
        #except:
        #    return False
        return True
       
            
    def PredictImage(self, image):    
        if not self.initialized() or image is None:
            return None

        ### to do -> ensure matching size, either here or in the model
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))
        
        if self.PredictFullImage:
            try:
                pred = self.Predict(image)
            except:
                # OOM
                pred = self.predictTiled(image)
        else:
            pred = self.predictTiled(image)


        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return pred
    
    def predictTiled(self, image):
        pred = np.zeros(image.shape,dtype='uint8')
        p_width = self.augmentation.outputwidth
        p_height = self.augmentation.outputheight
        for w in range (0,image.shape[1],p_width):
            for h in range (0,image.shape[0],p_height):
                
                w_0 = max(w -self.split_factor//2, 0)
                h_0 = max(h -self.split_factor//2, 0)
                    
                h_til = min(h+p_height+self.split_factor//2, image.shape[0])
                w_til = min(w+p_width+self.split_factor//2,  image.shape[1])
                
                # take a full tile at the borders results in better segmentations
                if h_til == image.shape[0]:
                    h_0 = max(image.shape[0] - (p_height + self.split_factor//2), 0)
                if w_til == image.shape[1]:
                    w_0 = max(image.shape[1] - (p_width + self.split_factor//2), 0)
                
                patch = image[h_0:h_til, w_0:w_til]
                pred_patch = self.Predict(patch)
                h_end = min(h+p_height,image.shape[0])
                w_end = min(w+p_width,image.shape[1])
                pred[h:h_end, w:w_end] = pred_patch[h-h_0:patch.shape[0]-(h_til-h_end),w-w_0:patch.shape[1]-(w_til-w_end)]
        return pred
    
    def Predict(self, image):
        if len(image.shape) == 2:
            image = image[...,np.newaxis]

        pad = (self.split_factor - image.shape[0] % self.split_factor, self.split_factor - image.shape[1] % self.split_factor)
        pad = (pad[0] % self.split_factor, pad[1] % self.split_factor)            
        if pad != (0,0):
            image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0,0)),'constant', constant_values=(0, 0))

        image = self.data.preprocessImage(image)
        image = image[np.newaxis, ...]
        
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
        return pred


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
    
    def _getLoss(self):
        if self.Loss == dlLoss.focal:
            return dl_losses.focal_loss_function(binary = self.NumClasses() <= 2, weighted = self.useWeightedDistanceMap)

    def _getMetrics(self):
        if self.Metric == dlMetric.iou:
            return dl_metrics.iou_function(binary = self.NumClasses() <= 2, weighted = self.useWeightedDistanceMap)

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