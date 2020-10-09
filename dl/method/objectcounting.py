# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:15:28 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.method.pixelbasedprediction import PixelBasedPrediction
import dl.models.resnet50_SegNet as resnet50_SegNet
import dl.models.simple_SegNet as simple_SegNet
from skimage.feature import peak_local_max
import utils.shapes.Point as Point
import numpy as np
import cv2
from dl.loss.regression_losses import RegressionLosses
from dl.metric.regression_metrics import RegressionMetrics



class ObjectCounting(PixelBasedPrediction, LearningMode):
    def __init__(self, parent):
        PixelBasedPrediction.__init__(self,parent)  
        LearningMode.__init__(self, parent)
        self.type = dlMode.Object_Counting
        self.metric = RegressionMetrics(parent)
        self.loss = RegressionLosses(parent)
        

        self.kernel = None
        self.scalefactor = 1
        self.kernel_stdev = 3
        self.peak_val = 125
        
        
    def LoadLabel(self, filename, height, width):
        label = np.zeros((height, width, 1), np.uint8)
        points, bg = Point.loadPoints(filename)
        return Point.drawPointsToLabel(label, points, bg)
    
    def setObjectSize(self):
        # stdev should be between 1 and 10 and equals the expansion of an object
        # peakval is the max value of a single object at its coordinates
        k = cv2.getGaussianKernel(67, self.kernel_stdev) 
        self.kernel = k*np.transpose(k)
        self.scalefactor = self.peak_val/np.max(self.kernel)
        
    def prepreprocessLabel(self, label):
        # should we create heatmap here?
        return label
        
    def preprocessLabel(self, label):
        label = label.astype('float')
        self.setObjectSize()
        if self.parent.NumClasses > 2:
            # convert to an 1-hot encoded heatmap representing with the point coordinates in its maxima
            one_hot = np.eye(self.parent.NumClasses)[np.squeeze(label.astype(int), axis=-1)]
            for i in range(label.shape[0]):
                one_hot[i,...] = cv2.filter2D(one_hot[i,...],-1, self.kernel)
                one_hot[i,...,0] = one_hot[i,...,0]-(1-np.max(self.kernel))
                label = one_hot
        else:
            for i in range(label.shape[0]):
                label[i,...,0] = cv2.filter2D(label[i,...],-1, self.kernel)
            
        return label*self.scalefactor

    def getModel(self, nclasses, monochr):
        if self.parent.ModelType == 0:
            return simple_SegNet.simple_SegNet(nclasses, monochr, True)
        elif self.parent.ModelType == 1:    
            return resnet50_SegNet.resnet50_SegNet(nclasses, monochr, True)
        
    def extractShapesFromPrediction(self, prediction, unused, offset=(0,0)):
        return Point.extractPointsFromLabel(prediction, offset)
    
    def saveShapes(self, contours, path):
        Point.savePoints(contours, path)
        
    def convert2Image(self,prediction):
        prediction = np.squeeze(prediction)
        result = np.zeros(prediction.shape[0:2])
        # threshold?
        prediction[prediction < 1] = 0
        if self.parent.NumClasses > 2:
            # if we include zero channel we loose a lot of maxima
            maxarg = np.argmax(prediction, axis = 2)
            for i in range(1,prediction.shape[2]):
                peaks = peak_local_max(prediction[...,i], min_distance=3, indices = False)
                result[(maxarg==i) & peaks] = i
                
        else:
            peaks = peak_local_max(prediction, min_distance=3, indices = False)
            result[peaks] = 1
        return result
    
    def resizeLabel(self, label, shape):
        # every pixel != 0 will create exactly one pixel != 0 in the rezised image at the corresponding position
        newlabel = np.zeros((shape[1],shape[0]), dtype=label.dtype)
        if self.parent.NumClasses > 2:
            for i in range(1,np.max(label)+1):
                indices = np.where(label==i)
                idx_0 = (indices[0] / label.shape[0] * shape[1]).astype(int)
                idx_1 = (indices[1] / label.shape[1] * shape[0]).astype(int)
                newlabel[(idx_0, idx_1)] = i
        else:
            indices = np.where(label==1) 
            idx_0 = (indices[0] / label.shape[0] * shape[1]).astype(int)
            idx_1 = (indices[1] / label.shape[1] * shape[0]).astype(int)

            newlabel[(idx_0, idx_1)] = 1
        return newlabel
    
    def LabelDistance(self,l1, l2):
        return Point.distance_p1_p2(l1.coordinates,l2.coordinates)
    
    def LoadShapes(self,filename):
        points, bg = Point.loadPoints(filename)
        return points
        