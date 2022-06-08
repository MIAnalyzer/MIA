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
from tensorflow.keras.models import load_model
import os
from dl.models.segmentation_models import Unet, FPN, Linknet, PSPNet, get_preprocessing
from dl.models.deeplab.deeplab import Deeplabv3, deeplab_preprocess, deeplab_backbones
from dl.models.segmentation_models.backbones.backbones_factory import Backbones

from enum import Enum


class PixelBasedPrediction(ABC):
    def __init__(self, parent):
        self.parent = parent

    def PredictImage(self, image):    
        if not self.parent.initialized or image is None:
            return None


        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.parent.ImageScaleFactor), int(height*self.parent.ImageScaleFactor)))

        pred = self.PredictFullImage(image)
        pred = self.resizeLabel(pred, (width, height))
        return pred

    def PredictFullImage(self, image):
        if image.shape[0] * image.shape[1] < self.parent.augmentation.outputwidth * self.parent.augmentation.outputheight:
            return self.Predict(image)
        
        results = []
        origImage = image
        
        res = self.PredictFromGenerator(origImage)       
        results.append(res)
        
        if self.parent.tta:
            for i in range(5):               
                if i < 2:
                    flip = i
                    res = self.PredictFromGenerator(np.flip(origImage.copy(),flip))
                    results.append(np.flip(res,flip))
                else:
                    axes = (0,1)
                    iaxes = (1,0)
                    res = self.PredictFromGenerator(np.rot90(origImage.copy(), k = i-1, axes=axes))
                    results.append(np.rot90(res, k = i-1, axes=iaxes))


        results = np.stack(results, axis = 0)
        result = np.mean(results, axis = 0)
        return self.convert2Image(result).astype('uint8')

    def PredictFromGenerator(self, image):       
        generator = datagenerator.PredictionDataGenerator(self.parent, image)
        pred = self.parent.Model.predict(generator, workers=self.parent.worker)
        result = np.zeros(image.shape[0:2] + (self.parent.NumClasses,))
        sum_result = np.zeros(result.shape)
        
        border = self.parent.ModelOverlap//2
        p_width = self.parent.augmentation.outputwidth - 2*border
        p_height = self.parent.augmentation.outputheight - 2*border
        
        grad_patch = self.createModelOutputGradient()

        i = 0
        for h in range (0,image.shape[0],p_height):
            for w in range (0,image.shape[1],p_width):
                h_end = min(h+p_height,image.shape[0])
                w_end = min(w+p_width,image.shape[1])
                h_0 = min(h, result.shape[0]-p_height)
                w_0 = min(w, result.shape[1]-p_width)
        
                patch = pred[i]
                i += 1
                ### calculate the overlap with the help of grad_patch
                real_h0 = max(h_0-border,0)
                real_h_end = min(h_end+border,result.shape[0])
                real_w0 = max(w_0-border,0)
                real_w_end = min(w_end+border,result.shape[1])
                
                ph_0 = 0
                ph_end = patch.shape[0]
                pw_0 = 0
                pw_end = patch.shape[1]
                
                if real_h0 != h_0-border:
                    ph_0 = border - h_0
                if real_h_end != h_end+border:
                    ph_end = patch.shape[0] - (h_end+border - result.shape[0])
                
                if real_w0 != w_0-border:
                    pw_0 = border - w_0
                if real_w_end != w_end+border:
                    pw_end = patch.shape[1] - (w_end+border - result.shape[1])
                
                patch = patch * grad_patch
                
                result[real_h0:real_h_end, real_w0:real_w_end] += patch[ph_0:ph_end,pw_0:pw_end,...]
                sum_result[real_h0:real_h_end, real_w0:real_w_end] += (np.ones_like(patch) * grad_patch)[ph_0:ph_end,pw_0:pw_end]

                
                
        sum_result[sum_result == 0] = 1
        # correct weighting
        result = result / sum_result
        return result

    
    def createModelOutputGradient(self):
        # creates an 2d-array with 1s in the center and decaying to 0 at the borders
        width = self.parent.augmentation.outputwidth
        height = self.parent.augmentation.outputheight
        cutoff = self.parent.ModelOverlap//2
        
        a = np.arange(width)
        b = np.minimum(a,a[::-1])
        c = np.sqrt(b.reshape(1,width)*b.reshape(width,1))
        outarray = c / c[width//2,cutoff]
        outarray[outarray>1] = 1
        outarray = cv2.resize(outarray, (width,height))
        outarray = outarray.reshape(width,height,1)
        
        return outarray
    
    def Predict(self, image):
        if len(image.shape) == 2:
            image = image[...,np.newaxis]

        image = self.parent.data.preprocessImage(image)
        image = image[np.newaxis, ...]
        
        # convert_to_tensor is not necessary but improves performance because it avoids retracing
        pred = self.parent.Model.predict(tf.convert_to_tensor(self.parent.augmentation.checkModelSizeCompatibility(image),np.float32))
        pred = self.convert2Image(pred)

        pred = pred.astype('uint8')
        return pred

    def getImageSize4ModelInput(self, inputWidth, inputHeight):
        width = int(inputWidth*self.parent.ImageScaleFactor)
        height= int(inputHeight*self.parent.ImageScaleFactor)
        return width, height

    def setBackbone(self, backbone):
        self.backbone = backbone

    def setArchitecture(self, architecture):
        self.architecture = architecture

    def getBackbones(self):
        if self.architecture == 'Deeplabv3+':
            return deeplab_backbones()
        else:
            return Backbones.models_names()

    def getArchitectures(self):
        return ['UNet','FPN', 'Linknet', 'PSPNet', 'Deeplabv3+']

    def getPixelModel(self, nclasses, channels, activation):
        addextraInputLayer = False
        if self.pretrained:
            if self.architecture == 'Deeplabv3+':
                weights = 'pascal_voc'
                self.preprocessingfnc = deeplab_preprocess
            else:
                weights = 'imagenet'
                self.preprocessingfnc = get_preprocessing(self.backbone)
            inputsize = (None, None, 3)
            if channels != 3:
                addextraInputLayer = True 
        else:
            weights = None
            self.preprocessingfnc = None
            inputsize = (None, None, channels)


        if self.architecture == 'UNet':
            model = Unet(backbone_name=self.backbone, input_shape=inputsize, classes = nclasses, encoder_weights=weights, activation=activation)
        elif self.architecture == 'FPN':
            model = FPN(backbone_name=self.backbone, input_shape=inputsize, classes = nclasses, encoder_weights=weights, activation=activation)
        elif self.architecture == 'Linknet':
            model = Linknet(backbone_name=self.backbone, input_shape=inputsize, classes = nclasses, encoder_weights=weights, activation=activation)
        elif self.architecture == 'PSPNet':
            # must be divisible by 48
            width = self.parent.augmentation.outputwidth - self.parent.augmentation.outputwidth % 48
            height = self.parent.augmentation.outputheight - self.parent.augmentation.outputheight % 48
            # self.parent.augmentation.outputwidth = width
            # self.parent.augmentation.outputheight = height
            if self.pretrained:
                inputsize = (width, height, 3)
            else:
                inputsize = (width, height, channels)
            model = PSPNet(backbone_name = self.backbone, input_shape=inputsize, classes = nclasses, encoder_weights=weights, activation=activation)
        elif self.architecture == 'Deeplabv3+':
            if self.pretrained:
                # self.parent.augmentation.outputwidth = 512
                # self.parent.augmentation.outputheight = 512
                inputsize = (512, 512, 3)
            else:
                inputsize = (self.parent.augmentation.outputwidth, self.parent.augmentation.outputheight, channels)
            model = Deeplabv3(weights=weights, input_shape=inputsize, classes=nclasses, backbone=self.backbone, OS=16, alpha=1., activation=activation)

        else:
            raise ValueError("not a supported network architecture")

        if addextraInputLayer:
            input_layer = tf.keras.layers.Input(shape=(None, None, channels))
            layer0 = tf.keras.layers.Conv2D(3, (1, 1))(input_layer) 
            out = model(layer0)
            model = tf.keras.models.Model(input_layer, out, name=model.name)
            
        self.setModelInputSize()
        return model
    
    def setModelInputSize(self):
        if self.architecture == 'Deeplabv3+':
            if self.pretrained:
                self.parent.augmentation.outputwidth = 512
                self.parent.augmentation.outputheight = 512
                
        elif self.architecture == 'PSPNet':
            # must be divisible by 48
            self.parent.augmentation.outputwidth = self.parent.augmentation.outputwidth - self.parent.augmentation.outputwidth % 48
            self.parent.augmentation.outputheight = self.parent.augmentation.outputheight - self.parent.augmentation.outputheight % 48
                
                
    
    def loadmodel(self, path):
        if self.pretrained:
            if self.architecture == 'Deeplabv3+':
                self.preprocessingfnc = deeplab_preprocess
            else:
                self.preprocessingfnc = get_preprocessing(self.backbone)
        else:
            self.preprocessingfnc = None
        self.setModelInputSize()
        return load_model(path)
    
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