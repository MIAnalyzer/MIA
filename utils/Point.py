# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:22:17 2020

@author: Koerber
"""


# PointContour_ID = 987654321
import numpy as np
import math
from utils.Shape import Shape, Shapes, PointContour_ID


class Points(Shapes):
    def __init__(self):
        super(Points,self).__init__(PointContour_ID)

        
    def ShapeAlreadyExist(self,s):
        return next((True for elem in self.shapes if all(elem.coordinates == s.coordinates)), False)
    
    def load(self, filename):
        self.shapes = loadPoints(filename)
        
    def save(self, filename):
        savePoints(self.shapes, filename)
        
    def getShapeOfClass_x(self,x, point, distance):
        if self.empty():
            return None
        point = point[0]
        shapes = self.getShapesOfClass_x(x)
        if not shapes:
            return None
        dist_list = [distance_p1_p2(p.coordinates,point) for p in shapes]
        target = shapes[dist_list.index(min(dist_list))]
        if distance_p1_p2(target.coordinates,point) > distance:
            return None
        return target
      
    def getShape(self, point, distance):
        if self.empty():
            return None
        point = point[0]
        dist_list = [distance_p1_p2(p.coordinates,point) for p in self.shapes]
        target = self.shapes[dist_list.index(min(dist_list))]
        if distance_p1_p2(target.coordinates,point) > distance:
            return None
        return target
    

class Point(Shape):
    def __init__(self, classlabel, point):
        self.coordinates = point
        super(Point,self).__init__(classlabel, PointContour_ID)
        
    def inside(self, point, distance):
        if distance_p1_p2(self.coordinates,point) <= distance:
            return True
        else:
            return False
            

# utilities
def distance_p1_p2(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def savePoints(points, filename):
    f1 = lambda x: x.classlabel
    f2 = lambda x: x.coordinates
    pts = [f(x) for x in points for f in (f1, f2)]
    pts.insert(0,points[0].labeltype)
    np.savez(filename, *pts)
        
def loadPoints(filename):

    container = np.load(filename)
    data = [container[key] for key in container]
    
    ret = []
    if data[0] == PointContour_ID:
        label = data[1::2]
        array = data[2::2]
        ret = [Point(x,y) for x,y in zip(label,array)]
    return ret

