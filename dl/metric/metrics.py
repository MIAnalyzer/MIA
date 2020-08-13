# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:37:13 2020

@author: Koerber
"""

from abc import ABC, abstractmethod
from enum import Enum


class dlMetric(Enum):
    iou = 1
    dice = 2
    pixel_accuracy = 3
    mse = 4
    rmse = 5
    mae = 6
    msle = 7


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
        assert(metric in self.supportedmetrics)
        self._metric = metric