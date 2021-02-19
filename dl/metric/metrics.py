# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:37:13 2020

@author: Koerber
"""

from abc import ABC, abstractmethod
from enum import Enum


class dlMetric(Enum):
    iou = 1
    dice_coeff = 2
    pixel_accuracy = 3
    mean_squared_err = 4
    root_mean_squared_err = 5
    mean_abs_err = 6
    mean_squared_log_err = 7
    accuracy = 8
    precision = 9
    recall = 10
    f1 = 11


class Metrics(ABC):
    def __init__(self, parent):
        self.supportedmetrics = []
        self.parent = parent
        self._metric = None
        
    @abstractmethod
    def getMetric(self):
        pass
    
    @property
    def metric(self):
        return self._metric
    
    @metric.setter
    def metric(self, metric):
        self._metric = metric