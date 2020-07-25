# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:22:17 2020

@author: Koerber
"""


# PointContour_ID = 987654321
import numpy as np
import math
from utils.Shape import Shape, Shapes, PointContour_ID
from utils.Contour import findContours
import cv2


class Points(Shapes):
    def __init__(self):
        super(Points,self).__init__(PointContour_ID)
        self.mindistance = 3

    def addShapes(self,pts):
        for p in pts:
            self.addShape(p)

    def ShapeAlreadyExist(self,s):
        # return next((True for elem in self.shapes if all(elem.coordinates == s.coordinates)), False)
        return next((True for elem in self.shapes if distance_p1_p2(elem.coordinates,s.coordinates) < self.mindistance), False)
    
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
        self.coordinates = np.asarray(point).astype(int)
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

def extractPointsFromLabel(image):
    numclasses = np.max(image)
    points = []
    for i in range(1,numclasses+1):
        pts = np.where(image == i)
        p = zip(pts[1], pts[0])
        points.extend([Point(i, (x,y)) for x,y in p])

    return points


def extractPointsFromContours(image, minsize, offset = (0,0)):
    contours, _ = findContours(image, ext_only = True, offset=offset)
    points = []
    for c in contours:
        if cv2.contourArea(c) < minsize:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        points.append(Point(1, (cX, cY)))
    return points

def drawPointsToLabel(image, points):
    for p in points:
        image[p.coordinates[1],p.coordinates[0]] = p.classlabel      
    return image
