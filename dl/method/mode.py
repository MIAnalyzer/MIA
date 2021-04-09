# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:49:54 2020

@author: Koerber
"""

from abc import ABC, abstractmethod
from enum import Enum

class dlMode(Enum):
    Undefined = -1
    Classification = 0
    Object_Counting = 1
    Segmentation = 2
    # potentially at some point
    #Object_Detection = 3
    #Segmentation_3D = 4
    #Instance_Segmentation = 5

class dlPreprocess(Enum):
    scale = 0
    scale_shift = 1
    imagenet_mean = 2
    

class LearningMode(ABC):
    def __init__(self, parent):
        self.type = None
        self.parent = parent
        self.metric = None
        self.loss = None
        self.pretrained = True
        self.backbone = None
        self.architecture = None
        self.preprocessingfnc = None

    def preprocessImage(self, image):
        # should be set in case of using a pretrained model
        if self.preprocessingfnc:
            return self.preprocessingfnc(image)

        if self.parent.preprocess == dlPreprocess.imagenet_mean:
            image = image.astype('float')
            # bgr mean of imagenet dataset
            mean = [103.939, 116.779, 123.68]
            image[...,0] -= mean[0]
            image[...,1] -= mean[1]
            image[...,2] -= mean[2]
            return image
        if self.parent.preprocess == dlPreprocess.scale:
            return image.astype('float')/255
        if self.parent.preprocess == dlPreprocess.scale_shift:
            return image.astype('float')/127.5 -1
        
    @abstractmethod
    def LoadLabel(self, filename, height, width):
        pass
    
    @abstractmethod
    def prepreprocessLabel(self, label):
        # before training
        pass
        
    @abstractmethod
    def preprocessLabel(self, label):
        # during training
        pass

    @abstractmethod
    def setBackbone(self, bbname):
        return None

    @abstractmethod
    def setArchitecture(self, aname):
        return None
        
    @abstractmethod
    def getBackbones(self):
        return None

    @abstractmethod
    def getArchitectures(self):
        return None

    @abstractmethod
    def getModel(self, nclasses, channels):
        pass
    
    @abstractmethod
    def extractShapesFromPrediction(self, prediction, optArg1, optArg2):
        pass
    
    @abstractmethod
    def saveShapes(self, contours, path):
        pass
    
    @abstractmethod
    def PredictImage(self, image):
        pass
    
    @abstractmethod
    def resizeLabel(self, label, shape):
        pass
    
    @abstractmethod
    def countLabelWeight(self, label):
        pass

    @abstractmethod
    def getImageSize4ModelInput(self, inputWidth, inputHeight):
        pass
    
