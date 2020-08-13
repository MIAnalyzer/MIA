# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:22:54 2020

@author: Koerber
"""

from tensorflow.keras.layers import Activation, MaxPooling2D,BatchNormalization, Input, Flatten, Dense, Dropout, Conv2D, multiply, add, LeakyReLU, GlobalAveragePooling2D, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

def simple_SegNet(num_classes, monochrome = False, omitoutputfunc = False): 
    channels = 1 if monochrome is True else 3
    img_input = Input((None, None, channels))
    n_filters = 64
    if num_classes > 2:
        out_act_func = 'softmax'
        outputlayer = num_classes
    else :
        out_act_func = 'sigmoid'
        outputlayer = 1
    
    if omitoutputfunc == True:
        out_act_func = None
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