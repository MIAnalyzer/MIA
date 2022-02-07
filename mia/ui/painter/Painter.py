# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:37:52 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


from abc import ABC, abstractmethod
import numpy as np
import cv2
import utils.shapes.Contour as Contour
from ui.Tools import canvasTool
import random

class Painter(ABC):
    # every painter can draw contours 
    def __init__(self, canvas):
        self.canvas = canvas
        
        self.contours = Contour.Contours()
        self.NewContour = None
        # self.sketch = None
        self.tools = []
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.draw)
        self.tools.append(canvasTool.extend)
        self.tools.append(canvasTool.poly)
        self.tools.append(canvasTool.delete)
        self.trackChanges = []
        self.maxChanges = 4
        self.contoursketch = None
        self.mask = None
        self.objectcolors = []
    
    def clear(self):
        self.contours.clear()
        self.resetSmartMask()
        
    def resetSmartMask(self):
        self.mask = None
        
    @abstractmethod
    def load(self):
        pass
        
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def clearFOV(self):
        pass
    
    @abstractmethod
    def addShapes(self, shapes):
        pass
    
    @abstractmethod
    def checkForChanges(self):
        pass

    @abstractmethod
    def undo(self):
        pass

    def clearTracking(self):
        self.trackChanges.clear()

    def undoPossible(self):
        return len(self.trackChanges) >= 2

    @property
    def shapes(self):
        return self.contours
    
    @property
    def backgroundshapes(self):
        return self.contours.getShapesOfClass_x(0)
    
    @property
    def sketch(self):
        return self.contoursketch

    
    def draw(self):
        self.drawBackground()
    
    def getObjectColor(self, objnum):
        if len(self.objectcolors) < objnum:
            for i in range(0, objnum-len(self.objectcolors)):
                self.objectcolors.append(QColor(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        return self.objectcolors[objnum-1]

    def enableDrawBackgroundMode(self):
        self.tools.clear()
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.scale)
        self.tools.append(canvasTool.draw)
        self.tools.append(canvasTool.extend)
        self.tools.append(canvasTool.poly)
        self.tools.append(canvasTool.delete)

    def disableDrawBackgroundMode(self):
        self.tools.clear()
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.scale)
        
    def getPainter(self, color = None):
        if color is None:
            color = self.canvas.parent.ClassColor(self.canvas.parent.activeClass())
        # painter objects can only exist once per QWidget
        p = QPainter(self.canvas.image())
        color.setAlpha(255)    
        p.setPen(QPen(color, self.canvas.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))

        p.setBrush(QBrush(QColor(color),Qt.SolidPattern))
        p.setFont(QFont("Fixed",self.canvas.FontSize))
        return p
    
    def setColorTransparency(self, painter, color, transparency):
        if transparency == 1:        # opaque
            color.setAlpha(255)   
        elif transparency == -1:       # invisible
            color.setAlpha(0)
        elif transparency == 0:       # transparent
            color.setAlpha(self.canvas.ContourTransparency)
        painter.setBrush(QBrush(QColor(color),Qt.SolidPattern))
        

    def setPainterColor(self, painter, color):
        color.setAlpha(255)
        painter.setPen(QPen(color, self.canvas.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        painter.setBrush(QBrush(QColor(color),Qt.SolidPattern))

        
    def addline(self, p1 ,p2, color = None, dashed = False):
        p = self.getPainter(color)
        if dashed:
            pen = p.pen()
            pen.setStyle(Qt.DashDotDotLine)
            p.setPen(pen)
        else:
            pen = p.pen()
            pen.setStyle(Qt.SolidLine)
            p.setPen(pen)
        
        p.drawLine(p2, p1)
        self.canvas.updateImage()
        
    def addTriangle(self, point, size, color = None):
        psize = self.canvas.pen_size
        self.canvas.pen_size = 1
        p = self.getPainter(color)
        polygon = QPolygonF()
        polygon.append(point)
        point2 = QPointF(point.x() + size/2, point.y() - size)
        polygon.append(point2)
        point3 = QPointF(point.x() - size/2, point.y() - size)
        polygon.append(point3)
        p.drawPolygon(polygon)
        self.canvas.pen_size = psize
        self.canvas.updateImage()
        
    def addCircle(self, center, radius, color = None, erase = False):
        p = self.getPainter(color)
        p.setPen(QPen(Qt.NoPen))
        if erase:
            p.setBrush(QBrush(self.canvas.displayedimage.getImage()))
        p.drawEllipse(center, radius, radius)
        self.canvas.updateImage()


    def addRectangle(self, x, y, width, height, color = None):
        p = self.getPainter(color)
        p.drawRect(x, y, width, height)
        self.canvas.updateImage()


    #### contour drawing functions
    def drawcontour(self, contour):   
        painter = self.getPainter()
        path = self.contour2Path(contour)
        if not path.isEmpty():    
            if self.canvas.parent.colorCodeObjects() and contour.objectNumber != -1:
                color = self.getObjectColor(contour.objectNumber)
            else:
                color = self.canvas.parent.ClassColor(contour.classlabel)
            self.setPainterColor(painter, color)
            self.setColorTransparency(painter, color, transparency = 0)
            painter.drawPath(path)
            self.drawtrack(contour, painter, color)
            self.drawcontouraccessories(contour, painter)
            
            
    def drawBackground(self):    
        painter = self.getPainter()
        color = self.canvas.parent.ClassColor(0)
        self.setPainterColor(painter, color)
        bg = self.contours.getShapesOfClass_x(0)
        if len(bg) > 0:
            pm = self.canvas.displayedimage.getCanvas().copy()
            self.canvas.displayedimage.getCanvas().fill(QColor(0, 0, 0, self.canvas.ContourTransparency))
            painter.setCompositionMode(QPainter.CompositionMode_Source)
        
        for c in bg:  
            path = self.contour2Path(c)
            brush = QBrush(pm)             
            painter.fillPath(path,brush)
            self.setColorTransparency(painter, color,-1)
            painter.drawPath(path)

    def drawtrack(self, shape, painter, color):
        if self.canvas.parent.drawObjectTrack():
            track = self.canvas.parent.tracking.getObjectTrack(shape.objectNumber)
            if track and len(track) > 1:
                tp = self.canvas.parent.files.currentFrame if self.canvas.parent.tracking.stackMode else self.canvas.parent.files.currentImage
                track = track[:tp+1]
                path = QPainterPath()  
                fp = next(x[0] for x in enumerate(track) if x[1] != [-1,-1])
                path.moveTo(self.canvas.np2QPoint(track[fp]))
                [path.lineTo(self.canvas.np2QPoint(x)) for x in track[1:] if x != [-1,-1]]
                self.setColorTransparency(painter, color,-1)
                painter.drawPath(path)
                

    def contour2Path(self, contour):
        path = QPainterPath()
        if contour.points is not None:
            path.moveTo(self.canvas.np2QPoint(contour.points[0].reshape(2,1)))
            [path.lineTo(self.canvas.np2QPoint(x.reshape(2,1))) for x in contour.points]
            path.lineTo(self.canvas.np2QPoint(contour.points[0].reshape(2,1)))
            
            innerpath = QPainterPath()
            for c in contour.innercontours:    
                innerpath.moveTo(self.canvas.np2QPoint(c[0].reshape(2,1)))
                [innerpath.lineTo(self.canvas.np2QPoint(p.reshape(2,1))) for p in c]
                innerpath.lineTo(self.canvas.np2QPoint(c[0].reshape(2,1)))

            path = path.subtracted(innerpath)
        return path
    

    def addPoint2NewContour(self, p_image):
        p_last = self.canvas.np2QPoint(self.NewContour.getLastPoint())
        if (p_image != p_last):
            self.NewContour.addPoint(self.canvas.QPoint2np(p_image))
            self.addline(p_last,p_image)
            
  
    def prepareNewContour(self):
        if self.canvas.hasImage():
            width = self.canvas.image().width()
            height = self.canvas.image().height()

            self.contoursketch = np.zeros((height, width), np.uint8) 
            Contour.drawContoursToImage(self.contoursketch, self.contours.getShapesOfClass_x(self.canvas.parent.activeClass()))


    def getFinalContours(self):
        if self.sketch is None:
            return
            
        contours = Contour.extractContoursFromImage(self.contoursketch, not self.canvas.parent.allowInnerContours)
        # delete changed contours
        old_contours = self.contours.getShapesOfClass_x(self.canvas.parent.activeClass())
        changedcontours = Contour.getContoursNotinListOfContours(old_contours,contours)
        self.contours.deleteShapes(changedcontours)
            
        for c in contours:
            c.setClassLabel(self.canvas.parent.activeClass())
            
        self.contours.addShapes(contours)
        self.canvas.redrawImage()

        
    def finishNewContour(self, delete = False, close = True, drawaspolygon = False):
        if self.NewContour is not None:
            if close:
                self.NewContour.closeContour()
            i = 0 if delete else 1
            if self.NewContour.numPoints() > 1:
                if drawaspolygon:
                    cv2.polylines(self.sketch, [self.NewContour.points], 0, (i), thickness=max(self.canvas.pen_size,1), lineType =cv2.LINE_4)  
                else:
                    cv2.drawContours(self.sketch, [self.NewContour.points], 0, (i), -1) 
        self.getFinalContours()
        self.NewContour = None

    def drawcontouraccessories(self, contour, painter = None):
        color = self.canvas.parent.ClassColor(contour.classlabel)
        if painter is None:
            painter = self.getPainter(color = color)
        self.setPainterColor(painter, color)
        if self.canvas.drawShapeNumber:
            contournumber = self.contours.getShapeNumber(contour)
            painter.drawText(self.getLabelNumPosition(contour) , str(contournumber))
            
        if self.canvas.drawSkeleton:
            skeleton = contour.getSkeleton(self.canvas.skeletonsmoothingfactor)         
            
            skelpath = QPainterPath()
            if skeleton is not None:
                for cnt in skeleton:
                    i = 0
                    for c in cnt:         
                        if i == 0:
                            skelpath.moveTo(c[0][0], c[0][1])
                        else:
                            skelpath.lineTo(c[0][0], c[0][1])
                        i = i+1
                col = color.darker(65)
                self.setPainterColor(painter, col)
                self.setColorTransparency(painter, color, transparency = -1)
                painter.drawPath(skelpath)

        
    def getLabelNumPosition(self, contour):
        bp = contour.getBottomPoint()
        pos = QPoint(bp[0],bp[1])+QPoint(0,self.canvas.FontSize + 10)
        if pos.y() > self.canvas.image().height():
            pos.setY(self.canvas.image().height() - self.canvas.FontSize - 5)
        if pos.x() > self.canvas.image().width():
            pos.setX(self.canvas.image().width() - self.canvas.FontSize - 5)
        if pos.x() < 0:
            pos.setX(5)
        return pos