# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:15:28 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.method.pixelbasedprediction import PixelBasedPrediction
from skimage.feature import peak_local_max
import utils.shapes.Point as Point
from utils.shapes.Contour import removeBackground
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
        self.ignoreBackgroundLayer = True

        self.maxDetectionLimit = 5000

        self.architecture = 'UNet'
        self.backbone = 'resnet50'
        
    def LoadLabel(self, filename, height, width, rmBackground = False):
        label = np.zeros((height, width, 1), np.uint8)
        points, bg = Point.loadPoints(filename)
        label = Point.drawPointsToLabel(label, points, bg)
        if rmBackground:
            return removeBackground(label)
        return label
    
    def setObjectSize(self):
        k = cv2.getGaussianKernel(self.parent.od_kernel_size, -1)
        self.kernel = k*np.transpose(k)
        self.scalefactor = self.parent.od_peak_val/np.max(self.kernel)
        
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
            

            if self.ignoreBackgroundLayer:
                one_hot[...,0] = 0
            label = one_hot
        else:
            for i in range(label.shape[0]):
                label[i,...,0] = cv2.filter2D(label[i,...],-1, self.kernel)
            
        return label*self.scalefactor

    def getModel(self, nclasses, channels):
        return PixelBasedPrediction.getPixelModel(self, nclasses, channels, 'linear')

 
    def extractShapesFromPrediction(self, prediction, unused, offset=(0,0)):
        return Point.extractPointsFromLabel(prediction, offset)
    
    def saveShapes(self, contours, path):
        Point.savePoints(contours, path)
        
    def convert2Image(self,prediction):
        prediction = np.squeeze(prediction)
        # threshold?
        thresh = self.parent.od_peak_val//3
        calcResult = True
        # we need to add emergency break here, otherwise too many objects might be detected and break program
        while calcResult and thresh < self.parent.od_peak_val:
            result = np.zeros(prediction.shape[0:2])
            if self.parent.NumClasses > 2:
                maxarg = np.argmax(prediction, axis = 2)
                for i in range(1,prediction.shape[2]):
                    maxvals = peak_local_max(prediction[...,i], min_distance=max(2,self.parent.od_kernel_size//4), threshold_abs=thresh)
                    peaks = np.zeros_like(prediction[...,i], dtype='bool')
                    peaks[tuple(maxvals.T)] = True
                    result[(maxarg==i) & peaks] = i 
            else:
                maxvals = peak_local_max(prediction, min_distance=max(2,self.parent.od_kernel_size//3), threshold_abs=thresh)
                peak_mask = np.zeros_like(prediction, dtype='bool')
                peak_mask[tuple(maxvals.T)] = True  
                result[peak_mask] = 1
            calcResult = np.count_nonzero(result) > self.maxDetectionLimit
            thresh += 1

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
        
    def countLabelWeight(self, label):
        classValues = np.zeros((self.parent.NumClasses_real), dtype = np.float)
        for i in range(1,self.parent.NumClasses_real):
            classValues[i] = self.parent.od_kernel_stdev*np.count_nonzero((label == i))
        classValues[0] = label.size - self.parent.od_kernel_stdev*np.count_nonzero(label)
        return classValues