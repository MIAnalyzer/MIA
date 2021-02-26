# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:24:31 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.loss.classification_losses import ClassificationLosses
from dl.metric.classification_metrics import ClassificationMetrics
from utils.shapes.ImageLabel import ImageLabel, saveImageLabel, loadImageLabel, drawImageLabel
import dl.models.simple_ClassNet as simple_ClassNet
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

class Classification(LearningMode):
    def __init__(self, parent):
        super(Classification,self).__init__(parent)
        self.type = dlMode.Classification
        self.loss = ClassificationLosses(parent)
        self.metric = ClassificationMetrics(parent)
        
    def LoadLabel(self, filename, height, width):
        # we need an image here to perform background subtraction
        label = np.zeros((height, width, 1), np.uint8)
        imagelabel, bg = loadImageLabel(filename)
        return drawImageLabel(label, imagelabel, bg)
        
    def prepreprocessLabel(self, label):
        # convert image to 1D array
        return np.max(label).reshape(1)
    
    def preprocessLabel(self, label):
        if self.parent.NumClasses > 2:
            label = to_categorical(label, num_classes=self.parent.NumClasses)
        else:
            label = label.astype('float')
        return label

    def getModel(self, nclasses, monochr):
        return simple_ClassNet.simple_ClassNet(nclasses, monochr, True)

    def extractShapesFromPrediction(self, prediction, unused1, unused2):
        return None
    
    def saveShapes(self, contours, path):
        pass
    
    def PredictImage(self, image):
        pass
    
    def resizeLabel(self, label, shape):
        return cv2.resize(label, shape, interpolation = cv2.INTER_NEAREST)

    def countLabelWeight(self, label):
        return np.zeros((self.parent.NumClasses_real), dtype = np.float)