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
        self.supportedmetrics.append(dlMetric.mean_squared_err)
        self.supportedmetrics.append(dlMetric.root_mean_squared_err)
        self.supportedmetrics.append(dlMetric.mean_abs_err)
        self.supportedmetrics.append(dlMetric.mean_squared_log_err)
        self.metric = dlMetric.mean_abs_err
  
    def getMetric(self):
        if self.metric == dlMetric.mean_squared_err:
            return [mean_squared_error()]
        if self.metric == dlMetric.mean_abs_err:
            return [mean_absolute_error()]
        if self.metric == dlMetric.mean_squared_log_err:
            return [mean_squared_logarithmic_error()]
        if self.metric == dlMetric.root_mean_squared_err:
            return [root_mean_squared_error()]
        else:
            raise # not supported
            