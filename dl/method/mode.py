# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:49:54 2020

@author: Koerber
"""

from abc import ABC, abstractmethod
from enum import Enum

class dlMode(Enum):
    Classification = 0
    Object_Counting = 1
    Object_Tracking = 2
    Segmentation = 3
    Segmentation_Tracking = 4
    Segmentation_3D = 5
    Instance_Segmentation = 6
    Undefined = -1

class LearningMode(ABC):
    def __init__(self, parent):
        self.type = None
        self.parent = parent
        self.metric = None
        self.loss = None
        
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
    def getModel(self):
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

        
        
        