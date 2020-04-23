# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:23:06 2019

@author: Koerber
"""

import cv2
import numpy as np
import imutils
from PyQt5.QtCore import QPointF, QPoint
from skimage import morphology
from skimage.morphology import medial_axis, skeletonize
FreeFormContour_ID_legacy = 123456789
FreeFormContour_ID = 12345678910


BACKGROUNDCLASS = 255

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
    def __init__(self, classlabel, points = None, inner = None):
        self.classlabel = classlabel
        self.labeltype = FreeFormContour_ID
        self.points = None
        self.boundingbox = None # for performance issues saved as member var
        self.moments = None
        self.skeleton = None
        self.innercontours = []
        self.innerparams = []
        if isinstance(points, QPointF):
            self.points = [QPoint2np(points)]
        elif isinstance(points, np.ndarray):
            self.points = points
            if isinstance(inner, np.ndarray):
                for i in range(len(inner)):
                    self.innercontours.append(inner[i])
        elif points is None:
            self.points = None

    def __del__(self): 
        self.innercontours.clear()

    def setClassLabel(self,n):
        self.classlabel = n
            
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
        
    def getInnerContourParams(self):
        if not self.innerparams:
            self.innerparams = [ (cv2.moments(x),cv2.boundingRect(x)) for x in self.innercontours]
        return self.innerparams

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

        image = np.zeros((height, width), np.uint8)
#        image = np.zeros((height, width, 1), np.uint8)

        cv2.drawContours(image, [self.points - (l,t)], 0, (255), -1)
        blurred = cv2.medianBlur(image, 11)

        skel = morphology.skeletonize(blurred>0,  method='lee')
        
        # contour based length measurement
        if imutils.is_cv2() or imutils.is_cv4():
            contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif imutils.is_cv3():
            _, contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = contours[np.argmax([len(l) for l in contours])] + ([l,t])
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
        label[:] = (BACKGROUNDCLASS)
    else:
        for c in bg_contours:
            cnt = c.points
            if c.numPoints() > 0: 
                cv2.drawContours(label, [cnt], 0, (BACKGROUNDCLASS), -1)        
        
    for c in target_contours:
        cnt = c.points
        if c.numPoints() > 0: 
            cv2.drawContours(label, [cnt], 0, (int(c.classlabel)), -1)  
            cv2.drawContours(label, c.innercontours, -1, (BACKGROUNDCLASS), -1)
            # the contour of inner needs to be redrawn and belongs to outer, otherwise the inner contour is growing on each iteration
            cv2.drawContours(label, c.innercontours, -1, (int(c.classlabel)), 1)

    return label
       

def extractContoursFromLabel(image):
    image = np.squeeze(image).astype(np.uint8)
    image[np.where(image == BACKGROUNDCLASS)] = 0
    ret_contours = []
    maxclass = np.max(image)
    for i in range(1,maxclass+1):
        thresh = (image == i).astype(np.uint8)
        if imutils.is_cv2() or imutils.is_cv4():
            contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_CCOMP  , cv2.CHAIN_APPROX_SIMPLE)
        elif imutils.is_cv3():
            _, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            counter = -1
            for k, c in enumerate(contours):
                parent = hierarchy [0][k][3]
                if parent > -1:
                    ret_contours[counter].innercontours.append(c)
                else:
                    ret_contours.append(Contour(i,c))
                    counter += 1
    ret_contours.reverse()
    return ret_contours

def drawContoursToImage(image, contours): 
    for c in contours:
        cnt = c.points
        if c.numPoints() > 0: 
            cv2.drawContours(image, [cnt], 0, (1), -1) 
        cv2.drawContours(image, c.innercontours, -1, (0), -1) 
        # the contour of inner needs to be redrawn and belongs to outer, otherwise the inner contour is growing on each iteration
        cv2.drawContours(image, c.innercontours, -1, (1), 1) 

def extractContoursFromImage(image):
    image = np.squeeze(image).astype(np.uint8)
    ret_contours = []
    if imutils.is_cv2() or imutils.is_cv4():
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    elif imutils.is_cv3():
        _, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    if contours is not None:
        counter = -1
        for k,c in enumerate(contours):
            parent = hierarchy [0][k][3]
            if parent > -1:
                ret_contours[counter].innercontours.append(c)
            else:
                ret_contours.append(Contour(-1,c))
                counter += 1
    ret_contours.reverse()
    return ret_contours

def checkIfContourInListOfContours(contour, contours):
    # returns True if contour is contained in contours
    return next((True for elem in contours if (elem.getMoments() == contour.getMoments() and elem.getBoundingBox() == contour.getBoundingBox() and elem.getInnerContourParams() == contour.getInnerContourParams())), False)

def getContoursNotinListOfContours(contours1, contours2):
    # returns all contours in contours1 not in contours2
    c1 = [(x, x.getMoments(), x.getBoundingBox(), x.getInnerContourParams() ) for x in contours1]
    c2 = [( x.getMoments(), x.getBoundingBox(), x.getInnerContourParams() ) for x in contours2]   
    return [x[0] for x in c1 if x[1:] not in c2]


def saveContours(contours, filename):
    if len(contours) == 0:
        return

    f1 = lambda x: x.classlabel
    f2 = lambda x: x.points
    f3 = lambda x: x.innercontours
    cnts = [f(x) for x in contours for f in (f1, f2, f3)]
    cnts.insert(0,contours[0].labeltype)
    np.savez(filename, *cnts)
        
def loadContours(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    
    ret = []
    if data[0] == FreeFormContour_ID_legacy:
        label = data[1::2]
        array = data[2::2]
        ret = [Contour(x,y) for x,y in zip(label,array)]
    elif data[0] == FreeFormContour_ID:
        label = data[1::3]
        array = data[2::3]
        inner = data[3::3]
        ret = [Contour(x,y,z) for x,y,z in zip(label,array,inner)]

    return ret


def QPoint2np(p):
    return np.array([(p.x(),p.y())], dtype=np.int32)
    
def np2QPoint(p):
    return QPointF(p[0],p[1])