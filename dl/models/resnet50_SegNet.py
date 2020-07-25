# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:22:27 2020

@author: Koerber
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Activation, MaxPooling2D,BatchNormalization, Input, Flatten, Dense, Dropout, Conv2D, multiply, add, LeakyReLU, GlobalAveragePooling2D, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from tensorflow.keras.models import Model

def resnet50_SegNet(num_classes, monochrome = False, addoutputfunc = True):
    if num_classes > 2:
        out_act_func = 'softmax'
        outputlayer = num_classes
    else :
        out_act_func = 'sigmoid'
        outputlayer = 1
           
    if addoutputfunc == False:
        out_act_func = None
    
    resnetmodel = ResNet50(include_top = False, weights='imagenet')
    x = resnetmodel.output

    #### to do: get skip_connections from resnetmodel

    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(512, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)

    x = (UpSampling2D((2, 2)))(x)
    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(256, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)

    x = (UpSampling2D((2, 2)))(x)
    x = (ZeroPadding2D((1, 1)))(x)
    x = (Conv2D(128, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    

    x = (UpSampling2D((2, 2)))(x)
    x = (ZeroPadding2D((1, 1) ))(x)
    x = (Conv2D(64, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    
    x = (UpSampling2D((2, 2)))(x)
    x = (ZeroPadding2D((1, 1) ))(x)
    x = (Conv2D(64, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)
    
    x = (UpSampling2D((2, 2)))(x)
    x = (ZeroPadding2D((1, 1) ))(x)
    x = (Conv2D(64, (3, 3), padding='valid'))(x)
    x = (BatchNormalization())(x)

    x = Conv2D(outputlayer, (3, 3),activation=out_act_func, padding='same')(x)


    return Model(inputs=resnetmodel.input, outputs=x)
