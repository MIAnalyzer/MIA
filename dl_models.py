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


def load_simple_model(num_classes, monochrome = False):
    
    channels = 1 if monochrome is True else 3
    img_input = Input((None, None, channels))
    n_filters = 64
    if num_classes > 2:
        out_act_func = 'softmax'
        outputlayer = num_classes
    else :
        out_act_func = 'sigmoid'
        outputlayer = 1
    
    conv_act = 'relu'
    regularizer = regularizers.l2(1e-4)
    

    conv1 = Conv2D(n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(img_input)

    conv1 = BatchNormalization()(conv1)  
    conv1 = Conv2D(n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(2 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(4 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(8 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(16 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])

    conv6 = Conv2D(8 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(8 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(4 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(2 * n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(n_filters, (3, 3), activation=conv_act, padding='same', kernel_regularizer = regularizer)(conv9)

    conv10 = Conv2D(outputlayer, (1, 1), activation=out_act_func)(conv9)
    model = Model(inputs=[img_input], outputs=[conv10])
    return model

def resnetSegModel(num_classes, monochrome = False):
    if num_classes > 2:
        out_act_func = 'softmax'
        outputlayer = num_classes
    else :
        out_act_func = 'sigmoid'
        outputlayer = 1
    
    resnetmodel = ResNet50(include_top = False, weights='imagenet')
    x = resnetmodel.output

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

    x = Conv2D(outputlayer, (3, 3),activation=out_act_func, padding='same')(x)


    return Model(inputs=resnetmodel.input, outputs=o)


    

def concat(x,y):
    ch, cw = get_crop_shape(K.int_shape(x), K.int_shape(y))
    y = Cropping2D(cropping=(ch, cw))(y)
    return Concatenate()([UpSampling2D(size=(2, 2))(x), y])

def get_crop_shape(target, refer):
    # width
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)