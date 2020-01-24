# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:23:06 2019

@author: Koerber
"""
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QPoint

FreeFormContour_ID = 123456789



class Contours():
    def __init__(self):
        self.contours = []
        self.labeltype = FreeFormContour_ID
        
    def addContour(self,c):
        #if c not in self.contours:
        if not checkIfContourInListOfContours(c, self.contours) and c.isValid():
            self.contours.append(c)
            
    def addContours(self,cts):
        new_cnts = getContoursNotinListOfContours(cts, self.contours)
        self.contours.extend(new_cnts)
            
    def deleteContour(self,c):
        self.contours.remove(c)
        
    def deleteContours(self, contours):
        for c in contours:
            self.deleteContour(c)
        
    def clear(self):
        self.contours.clear()
        
    def empty(self):
        return not self.contours
    
    def numOfContours(self):
        return len(self.contours)
    
    def loadContours(self, filename):
        self.contours = loadContours(filename)
        
    def saveContours(self, filename):
        saveContours(self.contours, filename)
        
    def getContoursOfClass_x(self, x):
        l = []
        for c in self.contours:
            if c.classlabel == x:
                l.append(c)
        return l        
    
    def getContour(self, point):
        for c in self.contours: 
            if c.inside(point):
                return c
            
    def getclosestContour(self, point, dist, label = None):
        for c in self.contours: 
            if -c.distance(point) < dist:
                if label is None:
                    return c
                elif label == c.classlabel:
                    return c
        return None


class Contour():
    def __init__(self, classlabel, points = None):
        self.classlabel = classlabel
        self.labeltype = FreeFormContour_ID
        self.points = None
        self.boundingbox = None # for performance issues saved as member var
        self.moments = None
        if isinstance(points, QPointF):
            self.points = [QPoint2np(points)]
        elif isinstance(points, np.ndarray):
            self.points = points
        elif points is None:
            self.points = None
            
    def addPoint(self, point):
        self.points = np.concatenate([self.points, [QPoint2np(point)]])
            
    def closeContour(self):
        self.points = np.concatenate([self.points, self.points[0].reshape(1,1,2)])
        
    def getFirstPoint(self):
        return np2QPoint(self.points[0][0])
    
    def getLastPoint(self):
        return np2QPoint(self.points[-1][0])
    
    def isValid(self):
        if len(self.points) > 3 and cv2.contourArea(self.points) > 3:    # <- hard limit no contours smaller 4 pixels allowed
            return True
        else:
            return False
        
    def getBoundingBox(self):
        if self.boundingbox is None:
            self.boundingbox = cv2.boundingRect(self.points)
        return self.boundingbox
    
    def getMoments(self):
        if self.moments is None:
            self.moments = cv2.moments(self.points)
        return self.moments
        
    def getSize(self):
        return cv2.contourArea(self.points)
    def numPoints(self):
        return len(self.points)
                    
    def inside(self, point):
        p = QPoint2np(point)
        inside = cv2.pointPolygonTest(self.points,  (p[0,0], p[0,1]), False) 
        return True if inside >= 0 else False
    
    def getBottomPoint(self):
        
        b = (self.points[self.points[..., 1].argmax()][0])
        return QPoint(b[0], b[1])
    
    def getTopPoint(self):
        b = (self.points[self.points[..., 1].argmin()][0])
        return QPoint(b[0], b[1])
    
    def distance(self, point):
        p = QPoint2np(point)
        return cv2.pointPolygonTest(self.points,  (p[0,0], p[0,1]), True) 
    
    def getCenter(self):
        M = cv2.moments(self.points)
        if M["m00"] == 0:
            raise Exception('division by zero')
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return QPoint(cX, cY)

        
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
    image = np.squeeze(image).astype(np.uint8)
    image[np.where(image == 255)] = 0
    ret_contours = []
    maxclass = np.max(image)

    for i in range(1,maxclass+1):
        thresh = (image == i).astype(np.uint8)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            for c in contours:
                ret_contours.append(Contour(i,c))

    return ret_contours

def drawContoursToImage(image, contours): 
    for c in contours:
        cnt = c.points
        if c.numPoints() > 0: 
            cv2.drawContours(image, [cnt], 0, (1), -1)  


def extractContoursFromImage(image):
    image = np.squeeze(image).astype(np.uint8)
    ret_contours = []
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        for c in contours:
            ret_contours.append(c)
    return ret_contours

def checkIfContourInListOfContours(contour, contours):
    # returns True if contour is contained in contours
    return next((True for elem in contours if (elem.getMoments() == contour.getMoments() and elem.getBoundingBox() == contour.getBoundingBox())), False)

def getContoursNotinListOfContours(contours1, contours2):
    # returns all contours in contours1 not in contours2
    c1 = [(x, x.getMoments(), x.getBoundingBox() ) for x in contours1]
    c2 = [( x.getMoments(), x.getBoundingBox() ) for x in contours2]   
    return [x[0] for x in c1 if x[1:] not in c2]


def saveContours(contours, filename):
    if len(contours) == 0:
        return
    cnts = []
    cnts.append(contours[0].labeltype)
    for c in contours:
        if c.isValid():
            cnts.append(c.classlabel)
            cnts.append(c.points)
    np.savez(filename, *cnts)
        
def loadContours(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    ret = []
    if data[0] != FreeFormContour_ID:
        return
    for i, arr in enumerate(data[1:]):
        if i % 2 == 0:
            label = arr
        else:
            c = Contour(label, arr)
            if c.isValid():
                ret.append(Contour(label, arr))
    return ret

def QPoint2np(p):
    return np.array([(p.x(),p.y())], dtype=np.int32)
    
def np2QPoint(p):
    return QPointF(p[0],p[1])