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
        self.supportedlosses.append(dlLoss.mse)
        self.supportedlosses.append(dlLoss.mae)
        self.supportedlosses.append(dlLoss.msle)
        self._loss = dlLoss.mse

    def getLoss(self):
        if self._loss == dlLoss.mse:
            return mean_squared_error()
        if self._loss == dlLoss.mae:
            return mean_absolute_error()
        if self._loss == dlLoss.msle:
            return mean_squared_logarithmic_error()
        else:
            raise # not supported
            

        
        