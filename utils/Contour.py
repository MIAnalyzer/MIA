# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:23:06 2019

@author: Koerber
"""

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QPoint
from skimage import morphology

FreeFormContour_ID = 123456789



class Contours():
    def __init__(self):
        self.contours = []
        self.minSize = 100
        self.labeltype = FreeFormContour_ID
        
    def addContour(self,c):
        if not checkIfContourInListOfContours(c, self.contours) and c.isValid(self.minSize):
            self.contours.append(c)
            
    def addContours(self,cts):
        cnts = getContoursNotinListOfContours(cts, self.contours)
        new_cnts = [ x for x in cnts if x.isValid(self.minSize)]
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
        cnts = loadContours(filename)
        self.contours = [ x for x in cnts if x.isValid(self.minSize)]
        
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
    
    def getContourNumber(self, c):
        return self.contours.index(c)+1
    


class Contour():
    def __init__(self, classlabel, points = None):
        self.classlabel = classlabel
        self.labeltype = FreeFormContour_ID
        self.points = None
        self.boundingbox = None # for performance issues saved as member var
        self.moments = None
        self.skeleton = None
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
    
    def isValid(self, minArea = 3):
        if len(self.points) >= 3 and cv2.contourArea(self.points) > minArea:   
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
    
    def getLeftPoint(self):
        b = (self.points[self.points[..., 0].argmin()][0])
        return QPoint(b[0], b[1])
    
    def getRightpPoint(self):
        b = (self.points[self.points[..., 0].argmax()][0])
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
    
    def getSkeletonLength(self):
        if self.skeleton is None:
            self.skeleton = self.getSkeleton()
        if self.skeleton is None:
            return 0
        else:
            return cv2.arcLength(self.skeleton, False)/2
    
    def getSkeleton(self):
        if self.skeleton is not None:
            return self.skeleton
        t = self.points[self.points[..., 1].argmin()][0][1]
        b = self.points[self.points[..., 1].argmax()][0][1]
        l = self.points[self.points[..., 0].argmin()][0][0]
        r = self.points[self.points[..., 0].argmax()][0][0]

        width = r-l
        height = b-t

        if width < 3 or height < 3:
            return None

        image = np.zeros((height, width, 1), np.uint8)
        cv2.drawContours(image, [self.points - (l,t)], 0, (255), -1)
        blurred = cv2.medianBlur(image, 11)

        skel = morphology.skeletonize(blurred>0,  method='lee')
        
        # contour based length measurement
        _, contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = contours[0] + ([l,t])
        self.skeleton = cv2.approxPolyDP(c, 3, False)
        
        return self.skeleton
    
        ## pixel based length measurement
#        diag_kernel = np.array(([-1, 1, -1],	[1, -10, 1],	[-1, 1, -1]), dtype="int")
#        diac = cv2.filter2D(skel,-1, diag_kernel)
#        diag_steps = np.count_nonzero(diac)//2 - 1
#        total = np.count_nonzero(skel)
#        length = total - diag_steps + diag_steps * np.sqrt(2)
#        pts = np.where(skel>0) 
#        x = pts[0] + t
#        y = pts[1] + l
#
#        return x,y

        
################# helper functions #################
def LoadLabel(filename, height, width):
    contours = loadContours(filename)
    label = np.zeros((height, width, 1), np.uint8)
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

    f1 = lambda x: x.classlabel
    f2 = lambda x: x.points
    cnts = [f(x) for x in contours if x.isValid() for f in (f1, f2)]   
    cnts.insert(0,contours[0].labeltype)
    np.savez(filename, *cnts)
        
def loadContours(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    
    if data[0] != FreeFormContour_ID:
        return
    
    label = data[1::2]
    array = data[2::2]
    ret = [Contour(x,y) for x,y in zip(label,array)]
    return ret


#def saveContours(contours, filename):
#    if len(contours) == 0:
#        return
#    cnts = []
#    cnts.append(contours[0].labeltype)
#    for c in contours:
#        if c.isValid():
#            cnts.append(c.classlabel)
#            cnts.append(c.points)
#    np.savez(filename, *cnts)
#        
#def loadContours(filename):
#    container = np.load(filename)
#    data = [container[key] for key in container]
#    ret = []
#    if data[0] != FreeFormContour_ID:
#        return
#    for i, arr in enumerate(data[1:]):
#        if i % 2 == 0:
#            label = arr
#        else:
#            c = Contour(label, arr)
#            if c.isValid():
#                ret.append(Contour(label, arr))
#    return ret

def QPoint2np(p):
    return np.array([(p.x(),p.y())], dtype=np.int32)
    
def np2QPoint(p):
    return QPointF(p[0],p[1])