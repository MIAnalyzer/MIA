# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:30:53 2020

@author: Koerber
"""


from abc import ABC, abstractmethod
from enum import Enum

class dlLoss(Enum):
    focal = 1
    cross_entropy = 2
    dice_coefficient = 3
    kl_divergence = 4
    mse = 5
    mae = 6
    msle = 7


class Losses(ABC):
    def __init__(self, parent):
        self.supportedlosses = []
        self.parent = parent
        self._loss = None
        
    @abstractmethod
    def getLoss(self):
        pass
    
    @property
    def loss(self):
        return self._loss
    
    @loss.setter
    def loss(self, loss):
        assert(loss in self.supportedlosses)
        self._loss = loss
        
    
    
