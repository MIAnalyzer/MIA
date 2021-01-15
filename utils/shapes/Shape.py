# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:01:53 2020

@author: Koerber
"""

from abc import ABC, abstractmethod
FreeFormContour_ID_legacy = 123456789
FreeFormContour_ID = 12345678910
PointContour_ID = 987654321
IMAGELABEL_ID = 1234554321
UNKNOWN = -1
import numpy as np

class Shapes(ABC):
    def __init__(self, labeltype):
        self.labeltype = labeltype
        self.shapes = []
        
                
    def getMaxClass(self):
        labels = [s.classlabel for s in self.shapes]
        if labels != []:
            return max(labels)
        else:
            return 0
        
    def addShape(self,s):
        if s.labeltype != self.labeltype:
            return
        if not self.ShapeAlreadyExist(s):
            self.shapes.append(s)
          
    def deleteShape(self,s):
        if s:
            self.shapes.remove(s)
       
    def clear(self):
        self.shapes.clear()

    def numOfShapes(self):
        return len(self.shapes)
    
    def getShapeNumber(self, s):
        idx = self.shapes.index(s)
        num = self.shapes[idx].objectNumber
        if num != -1:
            return num
        else:
            return idx +1

    def getShapeNumberX(self, shapenum):
        givenNumbers = [x.objectNumber for x in self.shapes if x.objectNumber != -1]
        try:
            if givenNumbers == []:
                return self.shapes[shapenum-1]
            else:
                return self.shapes[givenNumbers.index(shapenum)]
        except:
            return None

    def empty(self):
        return not self.shapes
    
    def getShapesOfClass_x(self, x):
        l = []
        for c in self.shapes:
            if c.classlabel == x:
                l.append(c)
        return l
    
    def getShapeOfClass_x(self,x, position, distance):
        shapes = self.getShapesOfClass_x(x)
        for s in shapes: 
            if s.inside(position, distance):
                return s
    
    def getShape(self, position, distance):
        for s in self.shapes: 
            if s.inside(position, distance):
                return s
    
    @abstractmethod
    def load(self, filename):
        pass
        
    @abstractmethod
    def save(self, filename):
        pass
         
    @abstractmethod
    def ShapeAlreadyExist(self, s):
        pass
    
    
    
class Shape(ABC):
    def __init__(self, classlabel, shapeid):

        if isinstance(classlabel, np.ndarray) and classlabel.size > 1:
            self.classlabel = classlabel[0]
            self.objectNumber = classlabel[1]
        else:
            self.classlabel = classlabel
            self.objectNumber = -1
        self.labeltype = shapeid

    
    def setClassLabel(self, n):
        self.classlabel = n
        
    def setObjectNumber(self, n):
        self.objectNumber = n

    @abstractmethod
    def getPosition(self):
        pass
        
    @abstractmethod
    def inside(self, point, maxdistance):
        pass
    
    
    