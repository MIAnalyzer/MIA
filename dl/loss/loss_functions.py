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
# see segmentation

## regression
def mean_squared_error(class_weights=None):
    if class_weights is None:
        class_weights = 1
        
    def _mean_squared_error(y_true, y_pred):
        mse = K.mean(K.square(y_pred - y_true)*class_weights, axis=-1)
        return mse
    return _mean_squared_error

def mean_absolute_error(class_weights=None):
    if class_weights is None:
        class_weights = 1

    def _mean_absolute_error(y_true, y_pred):
        mae = K.mean(K.abs(y_true - y_pred)*class_weights, axis=-1)
        return mae
    return _mean_absolute_error

def mean_squared_logarithmic_error(class_weights=None):
    if class_weights is None:
        class_weights = 1

    def _mean_squared_logarithmic_error(y_true, y_pred):
        # clip at 0 or (-1 + epsilon) ??
        y_pred = K.clip(y_pred, 0, None)
        y_true = K.clip(y_true, 0, None)
        msle = K.mean(K.square(K.log(y_true + 1) - K.log(y_pred + 1))*class_weights, axis=-1)
        return msle
    return _mean_squared_logarithmic_error


## segmentation
def focal_loss(usedistmap=False, class_weights = None, gamma=2):
    if class_weights is None:
        class_weights = .25
    # would be numerically more stable to use logits directly tf.nn.softmax_cross_entropy_with_logits and extend for focal and weights
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    def _focal_loss(ytrue, ypred):
        if usedistmap:
            weightmap = K.expand_dims(ytrue[...,-1], axis=3)
            ytrue = ytrue[...,:-1]
        else: 
            weightmap = 1

        epsilon = K.epsilon()
        ypred = K.clip(ypred, epsilon, 1. - epsilon)
        return -K.mean(class_weights * K.pow(1. - ypred, gamma) * ytrue * K.log(ypred) * weightmap, axis=-1)
    return _focal_loss

def focal_loss_binary(usedistmap=False, class_weights = None, gamma=2):
    if class_weights is None:
        class_weights = .25
    # would be numerically more stable to use logits directly tf.nn.sigmoid_cross_entropy_with_logits and extend for focal and weights
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    def _focal_loss_binary(y_true, y_pred):
        if usedistmap:
            weightmap = K.expand_dims(y_true[...,-1], axis=3)
            y_true = y_true[...,:-1]
        else:
            weightmap = 1

        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * class_weights
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        return -K.mean(K.sum(alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t) * weightmap, axis=1))

    return _focal_loss_binary


def jaccard_distance_loss(usedistmap=False, class_weights=None, smooth=100):
    if class_weights is None:
        class_weights = 1
    # from https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    def _jaccard_distance_loss(y_true, y_pred):
        if usedistmap:
            weightmap = K.expand_dims(y_true[...,-1], axis=3)
            y_true = y_true[...,:-1]
        else:
            weightmap = 1
            
        intersec = K.abs(y_true * y_pred) * weightmap
        intersection = K.sum(intersec, axis=-1)
        intersection_w = K.sum(intersec*class_weights, axis=-1)
        sum_ = K.sum((K.abs(y_true) + K.abs(y_pred))*weightmap, axis=-1)
        jac = (intersection_w + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
    return _jaccard_distance_loss


def jaccard_distance_loss_binary(usedistmap=False, class_weights=None, smooth=100):
    if class_weights is None:
        class_weights = .5

    def _jaccard_distance_loss_binary(y_true, y_pred):
        if usedistmap:
            weightmap = K.expand_dims(y_true[...,-1], axis=3)
            y_true = y_true[...,:-1]
        else:
            weightmap = 1
            
        # my personal interpretation of weighted binary jaccard,
        # not entirely sure if that makes sense in all cases -> check systematically
        intersec = K.abs(y_true * y_pred) * weightmap
        inter_nom = K.sum(intersec*class_weights, axis=-1)
        inter_denom = K.sum(intersec*(1-class_weights), axis=-1)
        sum_ = K.sum((class_weights*K.abs(y_true) + (1-class_weights)*K.abs(y_pred))*weightmap, axis=-1)
        jac = (inter_nom + smooth) / (sum_ - inter_denom + smooth)
        return (1 - jac) * smooth
    return _jaccard_distance_loss_binary


def kullback_Leibler_divergence_loss_weighted(usedistmap=False):   
    def _kullback_Leibler_divergence_loss_weighted(y_true, y_pred):
        # class_weights not supported atm
        if usedistmap:
            y_true = y_true[...,:-1]

        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
    
        # equivalent to             
        # kld = tf.keras.losses.KLDivergence()
        # return kld(y_true, y_pred)
    
    return _kullback_Leibler_divergence_loss_weighted


# for loading models trained with custom loss
get_custom_objects()['_mean_squared_error'] = mean_squared_error()
get_custom_objects()['_mean_absolute_error'] = mean_absolute_error()
get_custom_objects()['_mean_squared_logarithmic_error'] = mean_squared_logarithmic_error()

get_custom_objects()['_focal_loss'] = focal_loss()
get_custom_objects()['_focal_loss_binary'] = focal_loss_binary()
get_custom_objects()['_jaccard_distance_loss'] = jaccard_distance_loss()
get_custom_objects()['_kullback_Leibler_divergence_loss_weighted'] = kullback_Leibler_divergence_loss_weighted()

