# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:52:38 2020

@author: Koerber
"""


from dl.metric.metrics import Metrics, dlMetric
from dl.metric.metric_functions import *

class RegressionMetrics(Metrics):
    def __init__(self,parent):
        super(RegressionMetrics,self).__init__(parent)
        self.supportedmetrics.append(dlMetric.mse)
        self.supportedmetrics.append(dlMetric.rmse)
        self.supportedmetrics.append(dlMetric.mae)
        self.supportedmetrics.append(dlMetric.msle)
        self._metric = dlMetric.mae
  
    def getMetric(self):
        if self._metric == dlMetric.mse:
            return [mean_squared_error()]
        if self._metric == dlMetric.mae:
            return [mean_absolute_error()]
        if self._metric == dlMetric.msle:
            return [mean_squared_logarithmic_error]
        if self._metric == dlMetric.rmse:
            return [root_mean_squared_error()]
        else:
            raise # not supported
            