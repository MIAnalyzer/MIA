# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:45:19 2020

@author: Koerber
"""

import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

# classification metrcis
# ...


# regression         
def mean_squared_error():
    return tf.keras.metrics.MeanSquaredError()

def mean_absolute_error():
    return tf.keras.metrics.MeanAbsoluteError()

def mean_squared_logarithmic_error():
    return tf.keras.metrics.MeanSquaredLogarithmicError()

def root_mean_squared_error():
    return tf.keras.metrics.RootMeanSquaredError ()


# segmentation
def mean_iou_binary(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()))

def mean_iou_binary_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou_binary(y_true, y_pred)

def mean_iou(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    true_pixels = K.argmax(y_true, axis=-1) 
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    for i in range(nb_classes-1):
        true_labels = K.cast(K.equal(true_pixels, i), K.floatx())
        pred_labels = K.cast(K.equal(pred_pixels, i), K.floatx())
        intersection = K.sum(true_labels * pred_labels)
        union = K.sum(true_labels) + K.sum(pred_labels) - intersection
        res = (intersection + K.epsilon()) / (union + K.epsilon())
        iou.append(res)
    return K.mean(tf.stack(iou))


def mean_iou_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou(y_true, y_pred)


def dice_coefficient(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    true_pixels = K.argmax(y_true, axis=-1) 
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    for i in range(nb_classes-1):
        true_labels = K.cast(K.equal(true_pixels, i), K.floatx())
        pred_labels = K.cast(K.equal(pred_pixels, i), K.floatx())
        nom = 2.*K.sum(true_labels * pred_labels)
        denom = K.sum(true_labels) + K.sum(pred_labels)
        res = (nom + K.epsilon()) / (denom + K.epsilon())
        iou.append(res)
    return K.mean(tf.stack(iou))

def dice_coefficient_binary(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    nom = 2.*K.sum(y_true * y_pred)
    denom = K.sum(y_true) + K.sum(y_pred)
    return K.mean((nom + K.epsilon()) / (denom + K.epsilon()))

def dice_coefficient_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return dice_coeffient(y_true, y_pred)

def dice_coefficient_binary_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return dice_coeffient_binary(y_true, y_pred)

def pixel_accuracy_binary(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    return K.mean(tf.count_nonzero(ypred == ytrue))

def pixel_accuracy_binary_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return pixel_accuracy_binary(y_true, y_pred)

def pixel_accuracy(y_true, y_pred):
    true_pixels = K.argmax(y_true, axis=-1) 
    pred_pixels = K.argmax(y_pred, axis=-1)
    return K.mean(tf.count_nonzero(true_pixels == pred_pixels))

def pixel_accuracy_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return pixel_accuracy(y_true, y_pred)


# for loading models trained with custom metrics
get_custom_objects()['mean_iou'] = mean_iou
get_custom_objects()['mean_iou_weighted'] = mean_iou_weighted
get_custom_objects()['mean_iou_binary'] = mean_iou_binary
get_custom_objects()['mean_iou_binary_weighted'] = mean_iou_binary_weighted

get_custom_objects()['dice_coeffient'] = dice_coefficient
get_custom_objects()['dice_coeffient_binary'] = dice_coefficient_binary
get_custom_objects()['dice_coeffient_weighted'] = dice_coefficient_weighted
get_custom_objects()['dice_coeffient_binary_weighted'] = dice_coefficient_binary_weighted