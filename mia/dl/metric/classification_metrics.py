# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:43:11 2020

@author: Koerber
"""

from dl.metric.metrics import Metrics, dlMetric
from dl.metric.metric_functions import *

class ClassificationMetrics(Metrics):
    def __init__(self,parent):
        super(ClassificationMetrics,self).__init__(parent)
        self.supportedmetrics.append(dlMetric.accuracy)
        self.supportedmetrics.append(dlMetric.precision)
        self.supportedmetrics.append(dlMetric.recall)
        self.supportedmetrics.append(dlMetric.f1)
        self.metric = dlMetric.accuracy
        
    def getMetric(self):
        if self.metric == dlMetric.accuracy:
            return self.accuracy_function(binary = self.parent.NumClasses <= 2)
        if self.metric == dlMetric.precision:
            return [precision]
        if self.metric == dlMetric.recall:
            return [recall]
        if self.metric == dlMetric.f1:
            return [f1]
        else:
            raise # not supported


    def accuracy_function(self, binary):
        # return 'accuracy' 
        if binary:
            return [accuracy_binary()]
        else:
            return [accuracy()]