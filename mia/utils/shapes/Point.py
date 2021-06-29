# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:22:17 2020

@author: Koerber
"""


# PointContour_ID = 987654321
import numpy as np
import math
from utils.shapes.Shape import Shape, Shapes, PointContour_ID
from utils.shapes.Contour import findContours, packContours, unpackContours, drawbackgroundToLabel
import cv2


class Points(Shapes):
    def __init__(self):
        super(Points,self).__init__(PointContour_ID)
        self.mindistance = 3

    def addShapes(self,pts):
        # atm it is not checked whether pts are within self.mindistance
        ext = [x for x in pts if x.labeltype == self.labeltype and not self.ShapeAlreadyExist(x)]
        self.shapes.extend(ext)

    def ShapeAlreadyExist(self,s):
        return next((True for elem in self.shapes if distance_p1_p2(elem.coordinates,s.coordinates) < self.mindistance), False)
    
    def load(self, filename):
        self.shapes, _ = loadPoints(filename)
        
    def save(self, filename):
        savePoints(self.shapes, filename)
        
    def deleteShapes(self, points):
        for p in points:
            self.deleteShape(p)
        
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
        
    def getPosition(self):
        return (self.coordinates[0], self.coordinates[1])

    def inside(self, point, distance):
        if distance_p1_p2(self.coordinates,point) <= distance:
            return True
        else:
            return False
            

# utilities
def distance_p1_p2(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def savePoints(points, filename, background = None):
    bg = packContours(background)
    pts = []
    if len(points) > 0:   
        f1 = lambda x: np.array((x.classlabel, x.objectNumber),dtype=int)
        f2 = lambda x: x.coordinates
        labels = [f1(x) for x in points]
        coords = [f2(x) for x in points]
        pts = [points[0].labeltype, np.stack(labels), np.stack(coords)]
    else:
        if bg:
            pts = [PointContour_ID, [], []]
    if bg:
        pts.extend(bg)
    if pts:
        np.savez(filename, *pts)

    
def loadPoints(filename):
    container = np.load(filename)
    data = [container[key] for key in container] 
    ret = []
    bg = None
    if data[0] == PointContour_ID:
        label = data[1]
        array = data[2]
        ret = [Point(x,y) for x,y in zip(label,array)]
        
        if len(data) > 3:
            bg = unpackContours(data[3:])
            
    return ret, bg

def extractPointsFromLabel(image, offset):
    numclasses = np.max(image)
    points = []
    for i in range(1,numclasses+1):
        pts = np.where(image == i)
        p = zip(pts[1], pts[0])
        points.extend([Point(i, (x+offset[0],y+offset[1])) for x,y in p])

    return points


def extractPointsFromContours(image, minsize, offset = (0,0)):
    contours, _ = findContours(image, ext_only = True, offset=offset)
    points = []
    for c in contours:
        c = cv2.convexHull(c, False)
        if cv2.contourArea(c) < minsize:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        points.append(Point(1, (cX, cY)))
    return points

def drawPointsToLabel(image, points, bg_contours): 
    image = drawbackgroundToLabel(image, bg_contours)
    
    for p in points:
        image[p.coordinates[1],p.coordinates[0]] = p.classlabel      
    return image
