# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:32:20 2020

@author: Koerber
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects
from dl.metric.metric_functions import dice_coefficient, dice_coefficient_binary, dice_coefficient_weighted, dice_coefficient_binary_weighted


## classification

## regression
def mean_squared_error():
    return tf.keras.losses.MeanSquaredError()

def mean_absolute_error():
    return tf.keras.losses.MeanAbsoluteError()
    
def mean_squared_logarithmic_error():
    return tf.keras.losses.MeanSquaredLogarithmicError()


## segmentation
def focal_loss(ytrue, ypred, weights=1, gamma=2, alpha = .25):
    # would be numerically more stable to use logits directly tf.nn.softmax_cross_entropy_with_logits and extend for focal and weights
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    ypred /= K.sum(ypred, axis=-1, keepdims=True)
    epsilon = K.epsilon()
    ypred = K.clip(ypred, epsilon, 1. - epsilon)
    return -K.mean(alpha * K.pow(1. - ypred, gamma) * ytrue * K.log(ypred) * weights, axis=-1)

def focal_loss_weighted(ytrue, ypred, gamma=2, alpha = .25):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    ytrue = ytrue[...,:-1]
    return focal_loss(ytrue, ypred, weightmap, gamma, alpha)

def focal_loss_binary(ytrue, ypred,weights = 1, gamma=2, alpha = .25):
    # would be numerically more stable to use logits directly tf.nn.sigmoid_cross_entropy_with_logits and extend for focal and weights
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    pt_1 = tf.where(tf.equal(ytrue, 1), ypred, tf.ones_like(ypred))
    pt_0 = tf.where(tf.equal(ytrue, 0), ypred, tf.zeros_like(ypred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) * weights) -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0) * weights)

def focal_loss_binary_weighted(ytrue, ypred, gamma=2, alpha = .25):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    #ytrue = K.expand_dims(ytrue[...,0], axis=3)
    ytrue = ytrue[...,:-1]
    return focal_loss_binary(ytrue, ypred, weightmap, gamma, alpha)


def categorical_cross_entropy_loss(ytrue, ypred, weights = 1):
    # side note: in retina-net paper alpha = 0.75 gives best result for cce
    return focal_loss(ytrue, ypred, weights=1, gamma=0, alpha = 0.5)
    
def categorical_cross_entropy_loss_binary(ytrue, ypred, weights = 1):
    # side note: in retina-net paper alpha = 0.75 gives best result for cce
    return focal_loss_binary(ytrue, ypred, weights=1, gamma=0, alpha = 0.5)

def categorical_cross_entropy_weighted(ytrue, ypred):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    ytrue = ytrue[...,:-1]
    return categorical_cross_entropy_loss(ytrue, ypred, weightmap)

def categorical_cross_entropy_binary_weighted(ytrue, ypred):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    ytrue = ytrue[...,:-1]
    return categorical_cross_entropy_loss_binary(ytrue, ypred, weightmap)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def dice_coefficient_binary_loss(y_true, y_pred):
    return -dice_coefficient_binary(y_true, y_pred)

def dice_coefficient_weighted_loss(y_true, y_pred):
    return -dice_coefficient_weighted(y_true, y_pred)

def dice_coefficient_binary_weighted_loss(y_true, y_pred):
    return -dice_coefficient_binary_weighted(y_true, y_pred)

def kullback_Leibler_divergence_loss_weighted(y_true, y_pred):
    ytrue = ytrue[...,:-1]
    return tf.keras.losses.KLDivergence()


# for loading models trained with custom loss
get_custom_objects()['focal_loss'] = focal_loss
get_custom_objects()['focal_loss_binary'] = focal_loss_binary
get_custom_objects()['focal_loss_weighted'] = focal_loss_weighted
get_custom_objects()['focal_loss_binary_weighted'] = focal_loss_binary_weighted

get_custom_objects()['categorical_cross_entropy_loss'] = categorical_cross_entropy_loss
get_custom_objects()['categorical_cross_entropy_loss_binary'] = categorical_cross_entropy_loss_binary
get_custom_objects()['categorical_cross_entropy_weighted'] = categorical_cross_entropy_weighted
get_custom_objects()['categorical_cross_entropy_binary_weighted'] = categorical_cross_entropy_binary_weighted

get_custom_objects()['dice_coeffient'] = dice_coefficient
get_custom_objects()['dice_coeffient_binary'] = dice_coefficient_binary
get_custom_objects()['dice_coeffient_weighted'] = dice_coefficient_weighted
get_custom_objects()['dice_coeffient_binary_weighted'] = dice_coefficient_binary_weighted


get_custom_objects()['kullback_Leibler_divergence_loss_weighted'] = kullback_Leibler_divergence_loss_weighted
