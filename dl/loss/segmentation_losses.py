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
        self.supportedlosses.append(dlLoss.focal)
        self.supportedlosses.append(dlLoss.cross_entropy)
        self.supportedlosses.append(dlLoss.dice_coefficient)
        self.supportedlosses.append(dlLoss.kl_divergence)
        self.loss = dlLoss.focal
        
    def getLoss(self):
        if self.loss == dlLoss.focal:
            return self.focal_Loss_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.useWeightedDistanceMap)
        if self.loss == dlLoss.cross_entropy:
            return dl_losses.categorical_cross_entropy_loss_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.useWeightedDistanceMap)
        if self.loss == dlLoss.dice_coefficient:
            return dl_losses.dice_coeffient_loss_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.useWeightedDistanceMap)
        if self.loss == dlLoss.kl_divergence:
            return dl_losses.kullback_Leibler_divergence_loss_function(weighted = self.parent.useWeightedDistanceMap)
        else:
            raise # not supported
                    
        
    def categorical_cross_entropy_loss_function(self, binary, weighted):
        if binary:
            if weighted:
                return categorical_cross_entropy_binary_weighted
            else:
                return categorical_cross_entropy_loss_binary
        else:
            if weighted:
                return categorical_cross_entropy_weighted
            else:
                return categorical_cross_entropy_loss 

    def focal_Loss_function(self, binary, weighted):
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
    
    def kullback_Leibler_divergence_loss_function(self, weighted):
        if weighted:
            return kullback_Leibler_divergence_loss_weighted
        else:
            return tf.keras.losses.KLDivergence()
    
    def dice_coeffient_loss_function(self, binary, weighted):
        if binary:
            if weighted:
                return dice_coeffient_binary_weighted
            else:
                return dice_coeffient_binary
        else:
            if weighted:
                return dice_coeffient_weighted
            else:
                return dice_coeffient


