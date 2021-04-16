# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:30:51 2020

@author: Koerber
"""

import tensorflow as tf
from dl.loss.losses import Losses, dlLoss
from dl.loss.loss_functions import *

class SegmentationLosses(Losses):
    def __init__(self,parent):
        super(SegmentationLosses,self).__init__(parent)
        self.supportedlosses.append(dlLoss.focal_loss)
        self.supportedlosses.append(dlLoss.cross_entropy)
        self.supportedlosses.append(dlLoss.dice_loss)
        self.supportedlosses.append(dlLoss.kl_divergence)
        self.loss = dlLoss.focal_loss
        
    def getLoss(self):
        if self.loss == dlLoss.focal_loss:
            return self.focal_Loss_function(binary = self.parent.NumClasses <= 2, usedist = self.parent.seg_useWeightedDistanceMap, class_weights=self.parent.data.class_weights)
        if self.loss == dlLoss.cross_entropy:
            return self.categorical_cross_entropy_loss_function(binary = self.parent.NumClasses <= 2, usedist = self.parent.seg_useWeightedDistanceMap, class_weights=self.parent.data.class_weights)
        if self.loss == dlLoss.dice_loss:
            return self.dice_coeffient_loss_function(binary = self.parent.NumClasses <= 2, usedist = self.parent.seg_useWeightedDistanceMap, class_weights=self.parent.data.class_weights)
        if self.loss == dlLoss.kl_divergence:
            return self.kullback_Leibler_divergence_loss_function(usedist = self.parent.seg_useWeightedDistanceMap)
        else:
            raise # not supported
                    
        
    def categorical_cross_entropy_loss_function(self, binary, usedist,class_weights):
        # side note: in retina-net paper alpha = 0.75 gives best result for cce
        if class_weights is None:
            class_weights = 0.5

        if binary:
            return focal_loss_binary(usedistmap=usedist, class_weights=class_weights, gamma=0)
        else:
            return focal_loss(usedistmap=usedist, class_weights=class_weights, gamma=0)

    def focal_Loss_function(self, binary, usedist, class_weights):
        if binary:
            return focal_loss_binary(usedistmap=usedist, class_weights=class_weights, gamma=2)
        else:
            return focal_loss(usedistmap=usedist, class_weights=class_weights, gamma=2)
    
    def kullback_Leibler_divergence_loss_function(self, usedist):
        return kullback_Leibler_divergence_loss_weighted(usedist)
    
    def dice_coeffient_loss_function(self, binary, usedist, class_weights):
        if binary:
            return jaccard_distance_loss_binary(usedistmap=usedist, class_weights=class_weights)
        else:
            return jaccard_distance_loss(usedistmap=usedist, class_weights=class_weights)
