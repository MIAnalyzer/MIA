# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:39:49 2020

@author: Koerber
"""

import tensorflow as tf
from dl.metric.metrics import Metrics, dlMetric
from dl.metric.metric_functions import *

class SegmentationMetrics(Metrics):
    def __init__(self,parent):
        super(SegmentationMetrics,self).__init__(parent)
        self.supportedmetrics.append(dlMetric.iou)
        self.supportedmetrics.append(dlMetric.dice_coeff)
        self.supportedmetrics.append(dlMetric.pixel_accuracy)
        self.metric = dlMetric.iou
        
    def getMetric(self):
        if self.metric == dlMetric.iou:
            return self.mean_iou_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.seg_useWeightedDistanceMap)
        if self.metric == dlMetric.dice_coeff:
            return self.dice_coefficient_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.seg_useWeightedDistanceMap)
        if self.metric == dlMetric.pixel_accuracy:
            return self.pixel_accuracy_function(binary = self.parent.NumClasses <= 2, weighted = self.parent.seg_useWeightedDistanceMap)
        else:
            raise # not supported
            
    def mean_iou_function(self, binary, weighted):
        #return [] # <- not running atm
        if binary:
            if weighted:
                return [mean_iou_binary_weighted]
            else:
                return [mean_iou_binary]
        else:
            if weighted:
                return [mean_iou_weighted]
            else:
                return [mean_iou]
         
    def dice_coefficient_function(self, binary, weighted):
        if binary:
            if weighted:
                return [dice_coefficient_binary_weighted]
            else:
                return [dice_coefficient_binary]
        else:
            if weighted:
                return [dice_coefficient_weighted]
            else:
                return [dice_coefficient]
        
    def pixel_accuracy_function(self, binary, weighted):
        if binary:
            if weighted:
                return [pixel_accuracy_binary_weighted]
            else:
                return [pixel_accuracy_binary]
        else:
            if weighted:
                return [pixel_accuracy_weighted]
            else:
                return [pixel_accuracy]