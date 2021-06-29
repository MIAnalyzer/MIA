# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:30:54 2020

@author: Koerber
"""

from dl.loss.losses import Losses, dlLoss
from dl.loss.loss_functions import *

class RegressionLosses(Losses):
    def __init__(self,parent):
        super(RegressionLosses,self).__init__(parent)
        self.supportedlosses.append(dlLoss.mean_squared_err)
        self.supportedlosses.append(dlLoss.mean_squared_log_err)
        self.supportedlosses.append(dlLoss.mean_abs_err)
        self.loss = dlLoss.mean_squared_err

    def getLoss(self):
        if self.loss == dlLoss.mean_squared_err:
            return mean_squared_error(class_weights = self.parent.data.class_weights)
        if self.loss == dlLoss.mean_abs_err:
            return mean_absolute_error(class_weights = self.parent.data.class_weights)
        if self.loss == dlLoss.mean_squared_log_err:
            return mean_squared_logarithmic_error(class_weights = self.parent.data.class_weights)
        else:
            raise # not supported
            

        
        