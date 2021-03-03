# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:40:27 2020

@author: Koerber
"""

import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
import dl.training.datagenerator as datagenerator
# import time

class PixelBasedPrediction(ABC):
    def __init__(self, parent):
        self.parent = parent
    
    def PredictImage(self, image):    
        if not self.parent.initialized or image is None:
            return None

        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.parent.ImageScaleFactor), int(height*self.parent.ImageScaleFactor)))

        pred = self.PredictFromGenerator(image)
        pred = self.resizeLabel(pred, (width, height))
        return pred

    def PredictFromGenerator(self, image):
        
        if image.shape[0] * image.shape[1] < self.parent.augmentation.outputwidth * self.parent.augmentation.outputheight:
            return self.Predict(image)
        generator = datagenerator.PredictionDataGenerator(self.parent, image)
        pred = self.parent.Model.predict(generator, workers=self.parent.worker)
        result = np.zeros(image.shape[0:2] + (self.parent.NumClasses,))
        i = 0
        border = self.parent.borderremoval//2
        p_width = self.parent.augmentation.outputwidth - 2*border
        p_height = self.parent.augmentation.outputheight - 2*border

        for h in range (0,image.shape[0],p_height):
            for w in range (0,image.shape[1],p_width):
                h_end = min(h+p_height,image.shape[0])
                w_end = min(w+p_width,image.shape[1])
                h_0 = min(h, result.shape[0]-p_height)
                w_0 = min(w, result.shape[1]-p_width)
        
                patch = pred[i]
                result[h_0:h_end, w_0:w_end] = patch[border:border+p_height,border:border+p_width,...]
                i += 1
        return self.convert2Image(result).astype('uint8')

    
    #def predictTiled(self, image):
    #    # deprecated, will be removed eventually
    #    pred = np.zeros(image.shape[0:2],dtype='uint8')
    #    p_width = self.parent.augmentation.outputwidth
    #    p_height = self.parent.augmentation.outputheight
    #    i = 0
        
    #    for h in range (0,image.shape[0],p_height):
    #        for w in range (0,image.shape[1],p_width):
    #            w_0 = max(w -self.parent.split_factor//2, 0)
    #            h_0 = max(h -self.parent.split_factor//2, 0)
                    
    #            # split factor need to be reworked and reasonable, get it from network
    #            h_til = min(h+p_height+self.parent.split_factor//2, image.shape[0])
    #            w_til = min(w+p_width+self.parent.split_factor//2,  image.shape[1])
                
    #            # a full tile at the borders results in better segmentations
    #            if h_til == image.shape[0]:
    #                h_0 = max(image.shape[0] - (p_height + self.parent.split_factor//2), 0)
    #            if w_til == image.shape[1]:
    #                w_0 = max(image.shape[1] - (p_width + self.parent.split_factor//2), 0)
                
    #            patch = image[h_0:h_til, w_0:w_til]
    #            pred_patch = self.Predict(patch)
    #            h_end = min(h+p_height,image.shape[0])
    #            w_end = min(w+p_width,image.shape[1])
                
    #            pred[h:h_end, w:w_end] = pred_patch[h-h_0:patch.shape[0]-(h_til-h_end),w-w_0:patch.shape[1]-(w_til-w_end),...]
    #            i += 1
    #    return pred
    
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

    def getImageSize4ModelInput(self, inputWidth, inputHeight):
        width = int(inputWidth*self.parent.ImageScaleFactor)
        height= int(inputHeight*self.parent.ImageScaleFactor)
        return width, height
    
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