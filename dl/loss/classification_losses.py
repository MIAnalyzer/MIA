# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:44:23 2020

@author: Koerber
"""

from dl.loss.losses import Losses, dlLoss
from dl.loss.loss_functions import *

class ClassificationLosses(Losses):
    def __init__(self,parent):
        super(ClassificationLosses,self).__init__(parent)

    def getLoss(self):
        pass