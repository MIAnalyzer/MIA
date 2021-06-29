# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:30:53 2020

@author: Koerber
"""


from abc import ABC, abstractmethod
from enum import Enum

class dlLoss(Enum):
    focal_loss = 1
    cross_entropy = 2
    dice_loss = 3
    kl_divergence = 4
    mean_squared_err = 5
    mean_abs_err = 6
    mean_squared_log_err = 7


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
        self._loss = loss
        
    
    
