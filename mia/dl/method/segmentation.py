# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:04:30 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.method.pixelbasedprediction import PixelBasedPrediction
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
        self.architecture = 'Unet'
        self.backbone = 'resnet50'

    def LoadLabel(self, filename, height, width, rmBackground = False):
        label = np.zeros((height, width, 1), np.uint8)
        contours = Contour.loadContours(filename)
        label = Contour.drawContoursToLabel(label, contours)
        if rmBackground:
            return Contour.removeBackground(label)
        return label
    
    def addWeightMap(self, label):
        if self.parent.seg_useWeightedDistanceMap:
            weights = dl_utils.createWeightedBorderMapFromLabel(label[np.newaxis,...], w0 = self.parent.seg_border_weight, sigma = self.parent.seg_border_dist)
            label = np.concatenate((label, weights[0,...]), axis = 2)
        return label
    
    def prepreprocessLabel(self, label):
        return self.addWeightMap(label)
        
    def preprocessLabel(self, mask):
        if self.parent.NumClasses > 2:
            if self.parent.seg_useWeightedDistanceMap:                 
                label = mask[...,0]
                weights = mask[...,1]
                label = to_categorical(label, num_classes=self.parent.NumClasses)
                mask = np.concatenate((label, weights[...,np.newaxis]), axis = 3) 
            else:
                mask = to_categorical(mask, num_classes=self.parent.NumClasses)
        else:
            mask = mask.astype('float')
        return mask

    def getModel(self, nclasses, channels):
        if nclasses > 2:
            out_activation = 'softmax'
        else:
            out_activation =  'sigmoid'
        return PixelBasedPrediction.getPixelModel(self, nclasses, channels, out_activation)

    def extractShapesFromPrediction(self, prediction, innercontours, offset=(0,0)):
        return Contour.extractContoursFromLabel(prediction, innercontours, offset)
    
    def saveShapes(self, contours, path):
        Contour.saveContours(contours, path)
        
    def convert2Image(self,prediction):
        prediction = np.squeeze(prediction)
        
        if self.parent.seg_wt_separation:
            return dl_utils.separatePredictions(prediction, self.parent.seg_wt_mindist, self.parent.seg_wt_thresh)
        
        if self.parent.NumClasses > 2:
            prediction = np.squeeze(np.argmax(prediction, axis = 2))
        else:
            prediction[prediction>0.5] = 1
            prediction[prediction<=0.5] = 0   
        return prediction

    def LabelDistance(self,l1, l2):
        return Contour.matchcontours(l1,l2)

    def LoadShapes(self,filename):
        contours = Contour.loadContours(filename)
        return contours
    
    def resizeLabel(self, label, shape):
        return cv2.resize(label, shape, interpolation = cv2.INTER_NEAREST)

    def countLabelWeight(self, label):
        classValues = np.zeros((self.parent.NumClasses_real), dtype = np.float)
        for i in range(self.parent.NumClasses_real):
            classValues[i] = np.count_nonzero((label == i))
        return classValues