# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:23:06 2019

@author: Koerber
"""
import cv2
import numpy as np
from PyQt5.QtCore import QPointF



class Contours():
    def __init__(self):
        self.contours = []
    def addContour(self,c):
        self.contours.append(c)
    def deleteContour(self,c):
        self.contours.remove(c)
    def clear(self):
        self.contours.clear()
    def empty(self):
        return not self.contours
    def numOfContours():
        return len(self.contours)
    def loadContours(self, filename):
        self.contours = loadContours(filename)
    def saveContours(self, filename):
        saveContours(self.contours, filename)
    def getContour(self, point):
        for c in self.contours: 
            if c.inside(point):
                return c
    def getclosestContour(self, point, dist):
        for c in self.contours: 
            if c.distance(point) < dist:
                return c

class Contour():
    def __init__(self, classlabel, points = None):
        self.classlabel = classlabel
        self.points = None
        if isinstance(points, QPointF):
            self.points = QPoint2np(points)
        elif isinstance(points, np.ndarray):
            self.points = points
        elif points is None:
            self.points = None
            
    def addPoint(self, point):
        self.points = np.concatenate([self.points, QPoint2np(point)])
            
    def closeContour(self):
        self.points = np.concatenate([self.points, self.points[0].reshape(1,2)])
        
    def getFirstPoint(self):
        return np2QPoint(self.points[0])
    
    def getLastPoint(self):
        return np2QPoint(self.points[-1])
    
    def isValid(self):
        if cv2.contourArea(self.points) > 1:
            return True
        else:
            return False
        
    def numPoints(self):
        return len(self.points)
                    
    def inside(self, point):
        p = QPoint2np(point)
        inside = cv2.pointPolygonTest(self.points,  (p[0,0], p[0,1]), False) 
        return True if inside >= 0 else False
    
    def distance(self, point):
        p = QPoint2np(point)
        return cv2.pointPolygonTest(self.points,  (p[0,0], p[0,1]), True) 

        
################# helper functions #################
def LoadLabel(filename, width, height):
    contours = loadContours(filename)
    label = np.zeros((width, height, 1), np.uint8)
    return drawContoursToLabel(label, contours)        
        
        
def drawContoursToLabel(label, contours):
    # split contours
    bg_contours, target_contours = [], []
    for c in contours:
       target = bg_contours if c.classlabel == 0 else target_contours
       target.append(c)
       
    if bg_contours == list():
        label[:] = (255)
    else:
        for c in bg_contours:
            cnt = c.points
            if c.numPoints() > 0: 
                cv2.drawContours(label, [cnt], 0, (255), -1)        
        
    for c in target_contours:
        cnt = c.points
        if c.numPoints() > 0: 
            # separate connecting contours
            cv2.drawContours(label, [cnt], 0, (255), 2)
            cv2.drawContours(label, [cnt], 0, (int(c.classlabel)), -1)  
    return label
       

def extractContoursFromLabel(image):
    image = image.astype(np.uint8)
#    _, thresh = cv2.threshold(image, 254, 255,  cv2.THRESH_BINARY)
#    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image[np.where(image == 255)] = 0
    ret_contours = []
    maxclass = np.max(image)
#    for c in contours:
#        #if cv2.contourArea(c)+1 <= (image.shape[0]-1)*(image.shape[1]-1):
#         ret_contours.append(Contour(0,c))
    for i in range(maxclass,0,-1):
        _, thresh = cv2.threshold(image, i-1, 255,  cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image[np.where(image == i)] = 0
        if contours is not None:
            for c in contours:
                ret_contours.append(Contour(i,c))

    return ret_contours


def saveContours(contours, filename):
    if len(contours) == 0:
        return
    cnts = []
    for c in contours:
        cnts.append(c.classlabel)
        cnts.append(c.points)
    np.savez(filename, *cnts)
        
def loadContours(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    ret = []
    for i, arr in enumerate(data):
        if i % 2 == 0:
            label = arr
        else:
            ret.append(Contour(label, arr))
    return ret


def QPoint2np(p):
    return np.array([(p.x(),p.y())], dtype=np.int32)
    
def np2QPoint(p):
    return QPointF(p[0],p[1])