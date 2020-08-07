# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:24:31 2020

@author: Koerber
"""

from dl.dl_mode import LearningMode, dlMode


class Classification(LearningMode):
    def __init__(self, parent):
        super(Classification,self).__init__(parent)
        self.type = dlMode.Classification
        
    def LoadLabel(self, filename, height, width):
        pass
        
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
    
    def resizeLabel(self, label, shape):
        return label