# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:14:06 2020

@author: Koerber
"""
from math import ceil
from enum import Enum
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


class lrMode(Enum):
    Constant = 0
    ReduceOnPlateau = 1
    Schedule = 2
    


class LearningRateSchedule():
    def __init__(self, parent):
        self.parent = parent
        self.ReduceEveryXEpoch = 25
        self.ReductionFactor = 0.5
        self.Mode = lrMode.Constant
        self.minlr = 0.000001
        
    def setMode(self, mode):
        self.Mode = mode
        if mode == lrMode.ReduceOnPlateau:
            self.ReduceEveryXEpoch = 5
            self.ReductionFactor = 0.2

        if mode == lrMode.Schedule:
            self.ReduceEveryXEpoch = 25
            self.ReductionFactor = 0.5

            
        
    def LearningRateSchedule(self):
        def schedule(epoch_idx):
            return max(self.minlr, self.parent.learning_rate * pow(self.ReductionFactor, epoch_idx // self.ReduceEveryXEpoch))
            
        return LearningRateScheduler(schedule=schedule)
    
    def ReduceOnPlateau(self):
        # optional monitor val_loss if train/test split is activated
        return ReduceLROnPlateau(monitor='loss', factor = self.ReductionFactor, patience=self.ReduceEveryXEpoch, min_lr = self.minlr)
                                 
    def LearningRateCallBack(self):
        if self.Mode == lrMode.Constant:
            return None
        if self.Mode == lrMode.ReduceOnPlateau:
            return self.ReduceOnPlateau()
        if self.Mode == lrMode.Schedule:
            return self.LearningRateSchedule()