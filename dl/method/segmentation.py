# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:04:30 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.method.pixelbasedprediction import PixelBasedPrediction
import dl.models.resnet50_SegNet as resnet50_SegNet
import dl.models.simple_SegNet as simple_SegNet
import utils.shapes.Contour as Contour
import dl.utils.dl_utils as dl_utils
import numpy as np
import cv2
from dl.loss.segmentation_losses import SegmentationLosses
from dl.metric.segmentation_metrics import SegmentationMetrics
from tensorflow.keras.utils import to_categorical

class Segmentation(PixelBasedPrediction, LearningMode):
    def __init__(self, parent):
        PixelBasedPrediction.__init__(self,parent)  
        LearningMode.__init__(self, parent)
        self.type = dlMode.Segmentation
        self.metric = SegmentationMetrics(parent)
        self.loss = SegmentationLosses(parent)
        self.useWeightedDistanceMap = False

    def LoadLabel(self, filename, height, width):
        label = np.zeros((height, width, 1), np.uint8)
        contours = Contour.loadContours(filename)
        return Contour.drawContoursToLabel(label, contours)
    
    def addWeightMap(self, label):
        if self.useWeightedDistanceMap:
            weights = dl_utils.createWeightedBorderMapFromLabel(label[np.newaxis,...])
            label = np.concatenate((label, weights[0,...]), axis = 2)
        return label
    
    def prepreprocessLabel(self, label):
        return self.addWeightMap(label)
        
    def preprocessLabel(self, mask):
        if self.parent.NumClasses > 2:
            if self.parent.useWeightedDistanceMap:                 
                label = mask[...,0]
                weights = mask[...,1]
                label = to_categorical(label, num_classes=self.parent.NumClasses)
                mask = np.concatenate((label, weights[...,np.newaxis]), axis = 3) 
            else:
                mask = to_categorical(mask, num_classes=self.parent.NumClasses)
        else:
            mask = mask.astype('float')
        return mask

    def getModel(self, nclasses, monochr):
        if self.parent.ModelType == 0:
            return simple_SegNet.simple_SegNet(nclasses, monochr, False)
        elif self.parent.ModelType == 1:    
            return resnet50_SegNet.resnet50_SegNet(nclasses, monochr, False)
        
    def extractShapesFromPrediction(self, prediction, innercontours, offset=(0,0)):
        return Contour.extractContoursFromLabel(prediction, innercontours, offset)
    
    def saveShapes(self, contours, path):
        Contour.saveContours(contours, path)
        
    def convert2Image(self,prediction):
        if self.parent.NumClasses > 2:
            prediction = np.squeeze(np.argmax(prediction, axis = 3))
        else:
            prediction = np.squeeze(prediction)
            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0
        return prediction
    
    def resizeLabel(self, label, shape):
        return cv2.resize(label, shape, interpolation = cv2.INTER_NEAREST)