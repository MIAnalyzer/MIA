# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:03:00 2020

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, SGD
from enum import Enum
from inspect import signature

class dlOptim(Enum):
    adam = 1
    adamax = 2
    adadelta = 3
    adagrad = 4
    SGDNesterov = 5


class Optimizer():
    def __init__(self,parent):
        self.parent = parent
        self.Optimizer = dlOptim.adam
        
    def getOptimizer(self):
        if self.Optimizer == dlOptim.adam:
            return Adam(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.adamax:
            return Adamax(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.adadelta:
            return Adadelta(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.adagrad:
            return Adagrad(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.SGDNesterov:
            return SGD(learning_rate=self.parent.learning_rate, momentum=0.9, nesterov=True) 
        else:
            raise # not supported
    
    def setOptimizer(self, opti):
        self.Optimizer = opti
        if self.Optimizer == dlOptim.adam:
            fun = Adam
        elif self.Optimizer == dlOptim.adamax:
            fun = Adamax
        elif self.Optimizer == dlOptim.adadelta:
            fun = Adadelta
        elif self.Optimizer == dlOptim.adagrad:
            fun = Adagrad
        elif self.Optimizer == dlOptim.SGDNesterov:
            fun = SGD  
        else:
            raise # not supported
        self.parent.learning_rate = signature(fun).parameters['learning_rate'].default
        