# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 08:58:49 2019

@author: Koerber
"""


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects


def focal_loss(ytrue, ypred, weights=1, gamma=2, alpha = .25):
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    ypred /= K.sum(ypred, axis=-1, keepdims=True)
    eps = K.epsilon()
    ypred = K.clip(ypred, eps, 1. - eps)
    return -K.sum(alpha * K.pow(1. - ypred, gamma) * ytrue * K.log(ypred) * weights, axis=-1)

def focal_loss_weighted(ytrue, ypred, gamma=2, alpha = .25):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    ytrue = ytrue[...,:-1]
    return focal_loss(ytrue, ypred, weightmap, gamma, alpha)

def focal_loss_binary(ytrue, ypred,weights = 1, gamma=2, alpha = .25):
    # similar as https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    pt_1 = tf.where(tf.equal(ytrue, 1), ypred, tf.ones_like(ypred))
    pt_0 = tf.where(tf.equal(ytrue, 0), ypred, tf.zeros_like(ypred))
    epsilon = K.epsilon()
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) * weights) -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0) * weights)

def focal_loss_binary_weighted(ytrue, ypred, gamma=2, alpha = .25):
    weightmap = K.expand_dims(ytrue[...,-1], axis=3)
    #ytrue = K.expand_dims(ytrue[...,0], axis=3)
    ytrue = ytrue[...,:-1]
    return focal_loss_binary(ytrue, ypred, weightmap, gamma, alpha)

def focal_loss_function(binary, weighted):
    if binary:
        if weighted:
            return focal_loss_binary_weighted
        else:
            return focal_loss_binary
    else:
        if weighted:
            return focal_loss_weighted
        else:
            return focal_loss


# for loading models trained with custom loss
get_custom_objects()['focal_loss'] = focal_loss
get_custom_objects()['focal_loss_binary'] = focal_loss_binary
get_custom_objects()['focal_loss_weighted'] = focal_loss_weighted
get_custom_objects()['focal_loss_binary_weighted'] = focal_loss_binary_weighted
