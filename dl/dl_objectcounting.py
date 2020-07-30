# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:15:28 2020

@author: Koerber
"""

from dl.dl_mode import LearningMode, dlMode
from dl.dl_pixelbasedprediction import PixelBasedPrediction
import dl.models.resnet50_SegNet as resnet50_SegNet
import dl.models.simple_SegNet as simple_SegNet
from skimage.feature import peak_local_max
import utils.Point as Point
import numpy as np
import cv2

class ObjectCounting(PixelBasedPrediction, LearningMode):
    def __init__(self, parent):
        super(ObjectCounting,self).__init__(parent)
        self.type = dlMode.Object_Counting
        
    def LoadLabel(self, filename, height, width):
        label = np.zeros((height, width, 1), np.uint8)
        points = Point.loadPoints(filename)
        return Point.drawPointsToLabel(label, points)
        
    def preprocessLabel(self, label):
        k = cv2.getGaussianKernel(7, 1.2)  ## <- kernel size and stdev should be user specific (as minimum object distance)
        kernel = k*np.transpose(k) 
        if self.parent.NumClasses > 2:
            # convert to an 1-hot encoded heatmap representing with the point coordinates in its maxima
            one_hot = np.eye(self.parent.NumClasses)[np.squeeze(label, axis=-1)]
            for i in range(label.shape[0]):
                one_hot[i,...] = cv2.filter2D(one_hot[i,...],-1, kernel)
                one_hot[i,...,0] = one_hot[i,...,0]-(1-np.max(kernel))
                label = one_hot
        else:
            for i in range(label.shape[0]):
                label[i,...,0] = cv2.filter2D(label[i,...],-1, kernel)

        # do we need to scale when using regression as output
        scale = 1
        return label*scale

    def getModel(self, nclasses, monochr):
        if self.parent.ModelType == 0:
            return simple_SegNet.simple_SegNet(nclasses, monochr, True)
        elif self.parent.ModelType == 1:    
            return resnet50_SegNet.resnet50_SegNet(nclasses, monochr, True)
        
    def extractShapesFromPrediction(self, prediction, unused):
        return Point.extractPointsFromLabel(prediction)
    
    def saveShapes(self, contours, path):
        Point.savePoints(contours, path)
        
    def convert2Image(self,prediction):
        prediction = np.squeeze(prediction)
        # threshold ?!
        prediction[prediction<0.075] = 0
        result = np.zeros(prediction.shape)
        if self.parent.NumClasses > 2:
            for i in range(1,result.shape[2]):
                peaks = peak_local_max(prediction[...,i], min_distance=2, indices = False)
                result[peaks] = i
        else:
            peaks = peak_local_max(prediction, min_distance=2, indices = False)
            result[peaks] = 1
        return result