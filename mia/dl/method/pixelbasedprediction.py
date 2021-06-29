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
            self.parent.augmentation.outputwidth = width
            self.parent.augmentation.outputheight = height
            if self.pretrained:
                inputsize = (width, height, 3)
            else:
                inputsize = (width, height, channels)
            model = PSPNet(backbone_name = self.backbone, input_shape=inputsize, classes = nclasses, encoder_weights=weights, activation=activation)
        elif self.architecture == 'Deeplabv3+':
            if self.pretrained:
                self.parent.augmentation.outputwidth = 512
                self.parent.augmentation.outputheight = 512
                inputsize = (512, 512, 3)
            else:
                inputsize = (self.parent.augmentation.outputwidth, self.parent.augmentation.outputheight, channels)
            model = Deeplabv3(weights=weights, input_shape=inputsize, classes=nclasses, backbone=self.backbone, OS=16, alpha=1., activation=activation)

        else:
            raise ValueError("not a supported network architecture")

        if addextraInputLayer:
            input = tf.keras.layers.Input(shape=(None, None, channels))
            layer0 = tf.keras.layers.Conv2D(3, (1, 1))(input) 
            out = model(layer0)
            model = tf.keras.models.Model(input, out, name=model.name)
        print(model.summary())
        return model
    
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