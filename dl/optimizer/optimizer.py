# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:03:00 2020

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, SGD
from enum import Enum

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
        print(self.Optimizer)
        if self.Optimizer == dlOptim.Adam:
            return Adam(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.AdaMax:
            return AdaMax(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.AdaDelta:
            return AdaDelta(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.AdaGrad:
            return Adagrad(learning_rate=self.parent.learning_rate)
        if self.Optimizer == dlOptim.SGDNesterov:
            return SGD(learning_rate=self.parent.learning_rate, momentum=0.9, nesterov=True) 
        else:
            raise # not supported
    
    def setOptimizer(self, opti):
        self.Optimizer = opti
        if self.Optimizer == dlOptim.Adam:
            fun = Adam
        elif self.Optimizer == dlOptim.AdaMax:
            fun = Adamax
        elif self.Optimizer == dlOptim.AdaDelta:
            fun = Adadelta
        elif self.Optimizer == dlOptim.AdaGrad:
            fun = Adagrad
        elif self.Optimizer == dlOptim.SGDNesterov:
            fun = SGD  
        else:
            raise # not supported
        self.parent.learning_rate = signature(fun).parameters['learning_rate'].default
        