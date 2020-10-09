# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:40:27 2020

@author: Koerber
"""

import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class PixelBasedPrediction(ABC):
    def __init__(self, parent):
        self.parent = parent
    
    def PredictImage(self, image):    
        if not self.parent.initialized or image is None:
            return None

        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.parent.ImageScaleFactor), int(height*self.parent.ImageScaleFactor)))
        
        pred = self.predictTiled(image)
        # pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        pred = self.resizeLabel(pred, (width, height))
        return pred
    
    def predictTiled(self, image):
        pred = np.zeros(image.shape[0:2],dtype='uint8')
        p_width = self.parent.augmentation.outputwidth
        p_height = self.parent.augmentation.outputheight
        i = 0
        for w in range (0,image.shape[1],p_width):
            for h in range (0,image.shape[0],p_height):
                w_0 = max(w -self.parent.split_factor//2, 0)
                h_0 = max(h -self.parent.split_factor//2, 0)
                    
                # split factor need to be reworked and reasonable, get it from network
                h_til = min(h+p_height+self.parent.split_factor//2, image.shape[0])
                w_til = min(w+p_width+self.parent.split_factor//2,  image.shape[1])
                
                # a full tile at the borders results in better segmentations
                if h_til == image.shape[0]:
                    h_0 = max(image.shape[0] - (p_height + self.parent.split_factor//2), 0)
                if w_til == image.shape[1]:
                    w_0 = max(image.shape[1] - (p_width + self.parent.split_factor//2), 0)
                
                patch = image[h_0:h_til, w_0:w_til]
                pred_patch = self.Predict(patch)
                h_end = min(h+p_height,image.shape[0])
                w_end = min(w+p_width,image.shape[1])
                
                pred[h:h_end, w:w_end] = pred_patch[h-h_0:patch.shape[0]-(h_til-h_end),w-w_0:patch.shape[1]-(w_til-w_end),...]
        return pred
    
    def Predict(self, image):
        if len(image.shape) == 2:
            image = image[...,np.newaxis]

        pad = (self.parent.split_factor - image.shape[0] % self.parent.split_factor, self.parent.split_factor - image.shape[1] % self.parent.split_factor)
        pad = (pad[0] % self.parent.split_factor, pad[1] % self.parent.split_factor)            
        if pad != (0,0):
            image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0,0)),'constant', constant_values=(0, 0))

        image = self.parent.data.preprocessImage(image)
        image = image[np.newaxis, ...]
        
        # convert_to_tensor is not necessary but improves performance because it avoids retracing
        pred = self.parent.Model.predict(tf.convert_to_tensor(image,np.float32))
        pred = self.convert2Image(pred)

        pred = pred[0:pred.shape[0]-pad[0],0:pred.shape[1]-pad[1],...]
        pred = pred.astype('uint8')
        return pred
    
    @abstractmethod
    def convert2Image(self,prediction):
        pass
    
    @abstractmethod
    def resizeLabel(self, label, shape):
        pass
        
    @abstractmethod
    def LabelDistance(self,l1, l2):
        pass

    @abstractmethod
    def LoadShapes(self,filename):
        pass