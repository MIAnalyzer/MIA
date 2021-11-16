# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:25:49 2020

@author: Koerber
"""

from dl.method.mode import LearningMode, dlMode


class UndefinedMode(LearningMode):
    def __init__(self, parent):
        super(UndefinedMode,self).__init__(parent)
        self.type = dlMode.Undefined
        
    def LoadLabel(self, filename, height, width, rmBackground = False):
        return None
        
    def prepreprocessLabel(self, label):
        return label
    
    def preprocessLabel(self, label):
        return label

    def setBackbone(self, bbname):
        pass

    def setArchitecture(self, aname):
        pass

    def getBackbones(self):
        return None

    def getArchitectures(self):
        return None

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

    def getImageSize4ModelInput(self, inputWidth, inputHeight):
        return 1,1