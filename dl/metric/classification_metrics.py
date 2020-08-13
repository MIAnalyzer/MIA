# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:43:11 2020

@author: Koerber
"""

from dl.metric.metrics import Metrics, dlMetric
from dl.metric.metric_functions import *

class ClassificationnMetrics(Metrics):
    def __init__(self,parent):
        super(RegressionMetrics,self).__init__(parent)


    def getMetric(self):
        pass