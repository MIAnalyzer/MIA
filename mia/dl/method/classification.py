# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:24:31 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode
from dl.loss.classification_losses import ClassificationLosses
from dl.metric.classification_metrics import ClassificationMetrics
from utils.shapes.ImageLabel import ImageLabel, saveImageLabel, loadImageLabel, drawImageLabel, extractImageLabel
from utils.shapes.Contour import removeBackground
from tensorflow.keras.utils import to_categorical
from dl.models.classification_models.tfkeras import Classifiers
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

class Classification(LearningMode):
    def __init__(self, parent):
        super(Classification,self).__init__(parent)
        self.type = dlMode.Classification
        self.loss = ClassificationLosses(parent)
        self.metric = ClassificationMetrics(parent)
        self.preprocessingfnc = None
        self.backbone = 'resnet50'
        self.architecture = 'standard'
        
    def LoadLabel(self, filename, height, width, rmBackground = False):
        # we need an image here to perform background subtraction
        label = np.zeros((height, width, 1), np.uint8)
        imagelabel, bg = loadImageLabel(filename)
        label = drawImageLabel(label, imagelabel, bg)
        if rmBackground:
            return removeBackground(label)
        return label
        
    def prepreprocessLabel(self, label):
        # convert image to 1D array
        return np.max(label).reshape(1)
    
    def preprocessLabel(self, label):
        if self.parent.NumClasses > 2:
            label = to_categorical(label, num_classes=self.parent.NumClasses)
        else:
            label = label.astype('float')
        return label

    def setBackbone(self, backbone):
        self.backbone = backbone

    def setArchitecture(self, architecture):
        pass

    def getBackbones(self):
        return Classifiers.models_names()

    def getArchitectures(self):
        return ['standard']

    def getModel(self, n_classes, channels):
        backboneModel, self.preprocessingfnc = Classifiers.get(self.backbone)
        addextraInputLayer = False
        if self.pretrained:
            weights = 'imagenet'
            size = (None, None, 3)
            if channels != 3:
                addextraInputLayer = True                
        else:
            weights = None
            self.preprocessingfnc = None  
            size = (None, None,channels)


        if 'nasnet' in self.backbone:
            if self.pretrained:
                if self.backbone == 'nasnetmobile':
                    size = (224, 224,3)
                    self.parent.augmentation.outputwidth = 224
                    self.parent.augmentation.outputheight = 224
                if self.backbone == 'nasnetlarge':
                    size = (331, 331,3)
                    self.parent.augmentation.outputwidth = 331
                    self.parent.augmentation.outputheight = 331   
            else:
                size = (self.parent.augmentation.outputwidth, self.parent.augmentation.outputheight,channels)
            

        basemodel = backboneModel(input_shape=size, weights=weights, include_top=False)

        if addextraInputLayer:
            input = tf.keras.layers.Input(shape=(None, None, channels))
            layer0 = tf.keras.layers.Conv2D(3, (1, 1))(input) 
            out = basemodel(layer0)
            basemodel = tf.keras.models.Model(input, out, name=basemodel.name)

        if n_classes > 2:
            out_activation = 'softmax'
        else:
            out_activation =  'sigmoid'

        # we cut here and pool, to rebuild models like vgg exactly, fc layers (a 4096, 4096 and 1000 units in case of vgg) should be added
        x = tf.keras.layers.GlobalAveragePooling2D()(basemodel.output)
        output = tf.keras.layers.Dense(n_classes, activation=out_activation, dtype='float32')(x)
        self.setModelInputSize()
        return tf.keras.models.Model(inputs=[basemodel.input], outputs=[output])
    
    def setModelInputSize(self):
        if 'nasnet' in self.backbone:
            if self.pretrained:
                self.parent.augmentation.outputwidth = 224
                self.parent.augmentation.outputheight = 224

    def loadmodel(self, path):
        if self.pretrained:
            _, self.preprocessingfnc = Classifiers.get(self.backbone)
        else:
            self.preprocessingfnc = None
        self.setModelInputSize()
        return load_model(path)


    def extractShapesFromPrediction(self, prediction, unused1, unused2):
        return extractImageLabel(prediction)
    
    def saveShapes(self, contours, path):
        saveImageLabel(contours, path)
    
    def PredictImage(self, image):
        if not self.parent.initialized or image is None:
            return None

        width, height = self.getImageSize4ModelInput(None,None)
        image = cv2.resize(image, (width, height))
        
        results = []
        origImage = image
        
        pred = self.Predict(image)
        results.append(pred)
        
        if self.parent.tta:
            for i in range(5):               
                if i < 2:
                    flip = i
                    image = np.flip(origImage.copy(),flip)
                    res = self.Predict(image)
                    results.append(res)
                else:
                    axes = (0,1)
                    iaxes = (1,0)
                    image = np.rot90(origImage.copy(), k = i-1, axes=axes)
                    res = self.Predict(image)
                    results.append(res)

        results = np.stack(results, axis = 0)
        result = np.mean(results, axis = 0)

        if self.parent.NumClasses > 2:
            result = np.argmax(result)
        else:
            result[result>0.5] = 1
            result[result<=0.5] = 0  
        return result
    
    def Predict(self,image):
        image = self.parent.data.preprocessImage(image)
        image = image[np.newaxis, ...]
        return self.parent.Model.predict(self.parent.augmentation.checkModelSizeCompatibility(image))
        
    
    def resizeLabel(self, label, shape):
        return cv2.resize(label, shape, interpolation = cv2.INTER_NEAREST)

    def countLabelWeight(self, label):
        weights = np.zeros((self.parent.NumClasses_real), dtype = np.float)
        value = int(np.mean(label))
        weights[value] = 1
        return weights

    def getImageSize4ModelInput(self, inputWidth, inputHeight):
        width = self.parent.augmentation.outputwidth
        height = self.parent.augmentation.outputheight
        return width, height