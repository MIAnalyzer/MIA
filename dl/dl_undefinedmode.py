# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:25:49 2020

@author: Koerber
"""

from dl.dl_mode import LearningMode, dlMode


class UndefinedMode(LearningMode):
    def __init__(self, parent):
        super(UndefinedMode,self).__init__(parent)
        self.type = dlMode.Undefined
        
    def LoadLabel(self, filename, height, width):
        return None
        
    def preprocessLabel(self, label):
        return label


    def getModel(self, nclasses, monochr):
        return None

    def extractShapesFromPrediction(self, prediction, innercontours):
        return None
    
    def saveShapes(self, contours, path):
        pass
    
    def PredictImage(self, image):
        pass