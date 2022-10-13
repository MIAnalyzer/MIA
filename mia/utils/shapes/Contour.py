# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:23:06 2019

@author: Koerber
"""

import cv2
import numpy as np

import math

from skimage import morphology
from skimage.morphology import medial_axis, skeletonize

BACKGROUNDCLASS = 255
from utils.shapes.Shape import Shape, Shapes, FreeFormContour_ID, FreeFormContour_ID_legacy


class Contours(Shapes):
    def __init__(self):
        self.minSize = 3
        super(Contours,self).__init__(FreeFormContour_ID)
        
    def ShapeAlreadyExist(self,c):
        return checkIfContourInListOfContours(c, self.shapes) and c.isValid(self.minSize)
            
    def addShapes(self,cts):
        cnts = getContoursNotinListOfContours(cts, self.shapes)
        new_cnts = [ x for x in cnts if x.labeltype == self.labeltype and x.isValid(self.minSize)]
        self.shapes.extend(new_cnts)

    def deleteShapes(self, contours):
        for c in contours:
            self.deleteShape(c)

    def load(self, filename):
        cnts = loadContours(filename)
        self.shapes = [ x for x in cnts if x.isValid(self.minSize)]
        
    def save(self, filename):
        saveContours(self.shapes, filename)    

            
    def getclosestContour(self, point, dist, label = None):
        for c in self.shapes: 
            if -c.distance(point) < dist:
                if label is None:
                    return c
                elif label == c.classlabel:
                    return c
        return None


# freeform
class Contour(Shape):
    def __init__(self, classlabel, points = None, inner = None):
        super(Contour,self).__init__(classlabel, FreeFormContour_ID)
        self.points = None
        self.boundingbox = None # for performance issues saved as member var
        self.moments = None
        self.skeleton = None
        self.innercontours = []
        self.innerparams = []
        

        if isinstance(points, np.ndarray):
            if len(points.shape) == 2:
                self.points = [points]
            else:
                self.points = points
                if isinstance(inner, np.ndarray):
                    self.unpackInnerContours(inner)
                
        elif points is None:
            self.points = None

    def __del__(self): 
        self.innercontours.clear()

    def setClassLabel(self,n):
        self.classlabel = n
            
    def addPoint(self, point):
        self.points = np.concatenate([self.points, [point]])
        self.boundingbox = None
            
    def closeContour(self):
        self.points = np.concatenate([self.points, self.points[0].reshape(1,1,2)])
        
    def getFirstPoint(self):
        return self.points[0][0]
    
    def getLastPoint(self):
        return self.points[-1][0]

    def isValid(self, minArea = 3):
        if len(self.points) >= 3 and cv2.contourArea(self.points) > minArea:   
            return True
        else:
            return False

    def addInnerContour(self, c):
        if len(c) >= 3 and cv2.contourArea(c) > 3:
            self.innercontours.append(c)
            
    def packedInnerContours(self):
        if len(self.innercontours) < 1:
            return self.innercontours
        else:
            # we pack a [-1,-1] at front of every contour
            # warning: bracket overload!!! :)          
            return np.concatenate([np.concatenate((np.array([[[-1,-1]]]),x)) for x in self.innercontours])
        
    def unpackInnerContours(self, inner_packed):
        if len(inner_packed) == 0:
            return
        # we split the array at the position of every [-1,-1] into separate contours and remove [-1,-1]
        separate = np.where(inner_packed == ([-1,-1]))
        s = separate[0][0::2]
        inner = np.split(inner_packed, s[1:])    
        self.innercontours = [x[1:,...] for x in inner]


    def getInnerContourParams(self):
        if not self.innerparams:
            self.innerparams = [ (cv2.moments(x),cv2.boundingRect(x)) for x in self.innercontours]
        return self.innerparams

    def getBoundingBox(self):
        if self.boundingbox is None:
            if self.numPoints() == 1:
                return self.points[0][0][0], self.points[0][0][1], 0,0
            else:
                self.boundingbox = cv2.boundingRect(self.points)
        return self.boundingbox
    
    def getMoments(self):
        if self.moments is None:
            self.moments = cv2.moments(self.points)
        return self.moments
        
    def getSize(self):
        # careful: contourarea uses Green formula
        # as we use this in segmentation non_zero pixels might be more accurate
        # consider changing to non_zero pixels
        outer = cv2.contourArea(self.points)
        inner = [cv2.contourArea(x) for x in self.innercontours]
        return outer - sum(inner)
    
    def numPoints(self):
        return len(self.points)
                    
    def inside(self, point, distance=0):
        # distance ignored atm, if needed use cv2.pointPolygonTest(self.points,  p, True) 
        p = (point[0,0], point[0,1])
        inside = cv2.pointPolygonTest(self.points,  p, False)   
        if inside < 0:
            return False
        if not self.innercontours:
            return True    
        else:
            return next((False for c in self.innercontours if cv2.pointPolygonTest(c,  p, False) > 0), True)

    def getBottomPoint(self):
        b = (self.points[self.points[..., 1].argmax()][0])
        return (b[0], b[1])
    
    def getTopPoint(self):
        b = (self.points[self.points[..., 1].argmin()][0])
        return (b[0], b[1])
    
    def getLeftPoint(self):
        b = (self.points[self.points[..., 0].argmin()][0])
        return (b[0], b[1])
    
    def getRightpPoint(self):
        b = (self.points[self.points[..., 0].argmax()][0])
        return (b[0], b[1])
    
    def distance(self, point):
        return cv2.pointPolygonTest(self.points,  (point[0,0], point[0,1]), True) 

    def getPosition(self):
        return self.getCenter()
    
    def getCenter(self):
        M = cv2.moments(self.points)
        if M["m00"] == 0:
            raise Exception('division by zero')
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    
    def getPerimeter(self):
        return cv2.arcLength(self.points,True)
    
    def getSkeletonLength(self, smoothing):
        if self.skeleton is None:
            self.skeleton = self.getSkeleton(smoothing)
        if self.skeleton is None:
            return 0
        else:
            return sum([cv2.arcLength(c, True) for c in self.skeleton])/2
#            return cv2.arcLength(self.skeleton, False)/2
    
    def getSkeleton(self, smoothing):
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
        inner = [i - (l,t) for i in self.innercontours]
        cv2.drawContours(image, inner, -1, (0), -1)
        
        
        blurred = cv2.medianBlur(image, smoothing)

        skel = morphology.skeletonize(blurred>0,  method='lee')

        # contour based length measurement
        contours, _ = findContours(skel)

        self.skeleton = [cv2.approxPolyDP(c + ([l,t]), 1, True) for c in contours if len(c)>0]
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


# utilities
def drawcontour(image, contour, ignoreclasslabel=False, separateContours = False):
    contouring(image, [contour], classlabel=contour.classlabel, ignoreclasslabel=False, separateContours = False)


def drawcontours(image, contours, classlabel=1, ignoreclasslabel=False, separateContours = False):
    cnt_with_innercnts = [x for x in contours if x.innercontours != []]
    cnt_without_innercnts = [x for x in contours if x.innercontours == []]
    
    for c in cnt_with_innercnts:
        contouring(image, [c], classlabel, ignoreclasslabel, separateContours)
    contouring(image, cnt_without_innercnts, classlabel, ignoreclasslabel, separateContours)
    
def contouring(image, contours, classlabel=1, ignoreclasslabel=False, separateContours = False):
    
    cnt = [x.points for x in contours if x.numPoints() > 0]
    inner = []
    [inner.extend(x.innercontours) for x in contours if x.innercontours != []]
    ref_image = image.copy()
    if separateContours:
        for c in cnt:
            cv2.drawContours(image, [c], -1, (0), 3)  
            cv2.drawContours(image, [c], -1, (1), -1) 
    else:
        if ignoreclasslabel:
            cv2.drawContours(image, cnt, -1, (1), -1)  
            cv2.drawContours(image, inner, -1, (BACKGROUNDCLASS), -1)
            # the contour of inner needs to be redrawn and belongs to outer, otherwise the inner contour is growing on each iteration
            cv2.drawContours(image, inner, -1, (1), 1)
            image[image==BACKGROUNDCLASS] = ref_image[image==BACKGROUNDCLASS]

        elif classlabel != 0:
            cv2.drawContours(image, cnt, -1, (int(classlabel)), -1)  
            cv2.drawContours(image, inner, -1, (BACKGROUNDCLASS), -1)
            # the contour of inner needs to be redrawn and belongs to outer, otherwise the inner contour is growing on each iteration
            cv2.drawContours(image, inner, -1, (int(classlabel)), 1)
            image[image==BACKGROUNDCLASS] = ref_image[image==BACKGROUNDCLASS]
        else:
            cv2.drawContours(image, cnt, -1, (BACKGROUNDCLASS), -1)  
            cv2.drawContours(image, inner, -1, (0), -1)
            # the contour of inner needs to be redrawn and belongs to outer, otherwise the inner contour is growing on each iteration
            cv2.drawContours(image, inner, -1, (BACKGROUNDCLASS), 1)


def drawContoursToLabel(label, contours, drawbackground = True):
    if contours == []:
        return

    maxclass = max([x.classlabel for x in contours])
    for i in range(maxclass+1):
        class_cnts = filter(lambda x: x.classlabel == i, contours)
        if i== 0:
            label = drawbackgroundToLabel(label, list(class_cnts))
        else:
            drawcontours(label, list(class_cnts), classlabel=i)
        
    if not drawbackground:
        label[label==BACKGROUNDCLASS] = 0
    
    return label

def drawbackgroundToLabel(label, background):
    if not background or background == list():
        label[:] = (BACKGROUNDCLASS)
    else:
        drawcontours(label, background, classlabel=0)
    return label
       
def extractContoursFromLabel(image, ext_only = False, offset=(0,0)):
    image = np.squeeze(image).astype(np.uint8)
    ret_contours = []
    counter = -1

    # contours
    if np.all(image == BACKGROUNDCLASS):
        return []
    maxclass = np.max(image[image!=BACKGROUNDCLASS])
    for i in range(maxclass+1):
        if i == 0: # background
            if np.all(image != 0):
                continue
            else:
                thresh = (image == BACKGROUNDCLASS).astype(np.uint8)
        else:
            thresh = (image == i).astype(np.uint8)

        contours, hierarchy = findContours(thresh, ext_only, offset)
        if contours is not None:
            
            for k, c in enumerate(contours):
                parent = hierarchy [0][k][3]
                if parent > -1:
                    ret_contours[counter].addInnerContour(c)
                else:
                    ret_contours.append(Contour(i,c))
                    counter += 1
    ret_contours.reverse()
    return ret_contours

def drawContoursToImage(image, contours, separate = False):  
    drawcontours(image, contours, ignoreclasslabel=True, separateContours=separate)


def extractContoursFromImage(image, ext_only = False, offset = (0,0)):
    image = np.squeeze(image).astype(np.uint8)
    ret_contours = []
    contours, hierarchy = findContours(image, ext_only, offset)

    if contours is not None:
        counter = -1
        for k,c in enumerate(contours):
            parent = hierarchy [0][k][3]
            if parent > -1:
                ret_contours[counter].addInnerContour(c)
            else:
                ret_contours.append(Contour(-1,c))
                counter += 1
    ret_contours.reverse()
    return ret_contours

def packContours(contours):
    if not contours or len(contours) == 0:
        return None

    f1 = lambda x: np.array((x.classlabel, x.objectNumber),dtype=int)
    f2 = lambda x: x.points
#    f3 = lambda x: x.innercontours # we need to allow_pickle = True in np.load then
    f3 = lambda x: x.packedInnerContours()
    cnts = [f(x) for x in contours if x.isValid() for f in (f1, f2, f3)]
    cnts.insert(0,contours[0].labeltype)
    return cnts
    
def unpackContours(data):
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

def saveContours(contours, filename):
    cnts = packContours(contours)
    if cnts:
        np.savez(filename, *cnts)
        
def loadContours(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    return unpackContours(data)

def findContours(image, ext_only = False, offset=(0,0)):
    structure = cv2.RETR_EXTERNAL if ext_only else cv2.RETR_CCOMP

    # requires opencv 4 or greater, opencv 3 returns 3 params
    contours, hierarchy = cv2.findContours(image, structure , cv2.CHAIN_APPROX_SIMPLE, offset = offset)

    return contours, hierarchy

def removeBackground(image):
    image[image==BACKGROUNDCLASS] = 0
    return image

def checkIfContourInListOfContours(contour, contours):
    # returns True if contour is contained in contours
    return next((True for elem in contours if (elem.getMoments() == contour.getMoments() and elem.getBoundingBox() == contour.getBoundingBox() and elem.getInnerContourParams() == contour.getInnerContourParams())), False)

def getContoursNotinListOfContours(contours1, contours2):
    # returns all contours in contours1 not in contours2
    c1 = [(x, x.getMoments(), x.getBoundingBox(), x.getInnerContourParams() ) for x in contours1]
    c2 = [( x.getMoments(), x.getBoundingBox(), x.getInnerContourParams() ) for x in contours2]   
    return [x[0] for x in c1 if x[1:] not in c2]

def matchcontours(contour1, contour2):
    c1 = contour1.getCenter()
    c2 = contour2.getCenter()

    # potentially cv2.matchShapes(contour1.points, contour2.points, cv2.CONTOURS_MATCH_I1,0) based on Hu-moments
    # could be used additionally to compare, for now only the distance is calculated 
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def getContourMinIntensity(image, contour):
    mask = np.zeros(image.shape[0:2], dtype = np.uint8)
    cv2.drawContours(mask, contour.points, -1,255,-1)
    minval = np.min(image[mask>0])
    return minval

def getContourMeanIntensity(image, contour):
    mask = np.zeros(image.shape[0:2], dtype = np.uint8)
    cv2.drawContours(mask, contour.points, -1,255,-1)
    meanval = np.mean(image[mask>0])
    return meanval

def getContourMaxIntensity(image, contour):
    mask = np.zeros(image.shape[0:2], dtype = np.uint8)
    cv2.drawContours(mask, contour.points, -1,255,-1)
    maxval = np.max(image[mask>0])
    return maxval