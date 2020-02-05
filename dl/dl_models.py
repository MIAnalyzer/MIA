# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:28:06 2019

@author: Koerber
"""

import dl.models.resnet50_SegNet as resnet50_SegNet
import dl.models.simple_SegNet as simple_SegNet


class Models():
#    def __init__(self):
        
    def getModel(self, TrainingType, modelType, num_Classes, monochrome = False):
        if modelType == 0:
            return simple_SegNet.simple_SegNet(num_Classes, monochrome)
        elif modelType == 1:    
            return resnet50_SegNet.resnet50_SegNet(num_Classes, monochrome)