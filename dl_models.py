# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:28:06 2019

@author: Koerber
"""

from tensorflow.keras.layers import Activation, MaxPooling2D,BatchNormalization, Input, Flatten, Dense, Dropout, Conv2D, multiply, add, LeakyReLU, GlobalAveragePooling2D, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
import tensorflow.python.keras.backend as K
import scipy.io
from tensorflow.keras.applications.resnet50 import ResNet50

import resnet50_SegNet
import simple_SegNet



class Models():
#    def __init__(self):
        
    def getModel(self, TrainingType, modelType, num_Classes, monochrome = False):
        if modelType == 0:
            return simple_SegNet.simple_SegNet(num_Classes, monochrome)
        elif modelType == 1:    
            return resnet50_SegNet.resnet50_SegNet(num_Classes, monochrome)