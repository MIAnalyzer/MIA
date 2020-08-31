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

class Painter(ABC):
    # every painter can draw contours 
    def __init__(self, canvas):
        self.canvas = canvas
        
        self.contours = Contour.Contours()
        self.NewContour = None
        self.sketch = None
        self.tools = []
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.draw)
        self.tools.append(canvasTool.extend)
        self.tools.append(canvasTool.poly)
        self.tools.append(canvasTool.delete)

    
    def clear(self):
        self.contours.clear()
        
    @abstractmethod
    def load(self):
        pass
        
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def checkForChanges(self):
        pass
    
    @property
    def shapes(self):
        return self.contours
    
    @property
    def backgroundshapes(self):
        return self.contours.getShapesOfClass_x(0)
    
    def draw(self):
        self.drawBackground()
    
    def enableDrawBackgroundMode(self):
        self.tools.clear()
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.draw)
        self.tools.append(canvasTool.extend)
        self.tools.append(canvasTool.poly)
        self.tools.append(canvasTool.delete)

    def disableDrawBackgroundMode(self):
        self.tools.clear()
        
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
        
    def addCircle(self, center , radius, color = None):
        p = self.getPainter(color)
        p.drawEllipse(center, radius, radius)
        self.canvas.updateImage()


    #### contour drawing functions
    def drawcontour(self, contour):     
        painter = self.getPainter()
        path = self.contour2Path(contour)
        if not path.isEmpty():           
            color = self.canvas.parent.ClassColor(contour.classlabel)
            self.setPainterColor(painter, color)
            self.setColorTransparency(painter, color, transparency = 0)
            painter.drawPath(path)
            self.drawcontouraccessories(contour, painter)
            
    def drawBackground(self):    
        painter = self.getPainter()
        color = self.canvas.parent.ClassColor(0)
        self.setPainterColor(painter, color)
        bg = self.contours.getShapesOfClass_x(0)
        if len(bg) > 0:
            pm = self.canvas.displayedimage.pixmap().copy()
            self.canvas.displayedimage.pixmap().fill(QColor(0, 0, 0, 255)) 
            painter.setOpacity((255-self.canvas.ContourTransparency)/255)
            painter.drawPixmap(0, 0, pm)
        
        for c in bg:  
            path = self.contour2Path(c)
            brush = QBrush(pm)             
            painter.setOpacity(1)
            painter.fillPath(path,brush)
            self.setColorTransparency(painter, color,-1)
            painter.drawPath(path)
        
    
    def contour2Path(self, contour):
        path = QPainterPath()
        if contour.points is not None:
            for i in range(contour.numPoints()):
                if i == 0:
                    path.moveTo(self.canvas.np2QPoint(contour.points[i].reshape(2,1)))
                else:
                    path.lineTo(self.canvas.np2QPoint(contour.points[i].reshape(2,1)))
            path.lineTo(self.canvas.np2QPoint(contour.points[0].reshape(2,1)))
            
            innerpath = QPainterPath()
            for c in contour.innercontours:
                for i in range(len(c)):
                    if i == 0:
                        p = self.canvas.np2QPoint(c[i].reshape(2,1))
                        innerpath.moveTo(self.canvas.np2QPoint(c[i].reshape(2,1)))
                    else:
                        innerpath.lineTo(self.canvas.np2QPoint(c[i].reshape(2,1)))
                innerpath.lineTo(p)
            path = path.subtracted(innerpath)
        return path
    

    def addPoint2NewContour(self, p_image):
        p_last = self.canvas.np2QPoint(self.NewContour.getLastPoint())
        if (p_image != p_last):
            self.NewContour.addPoint(self.canvas.QPoint2np(p_image))
            self.addline(p_last,p_image)
            
    def prepareNewContour(self):
        width = self.canvas.image().width()
        height = self.canvas.image().height()
        self.sketch = np.zeros((height, width, 1), np.uint8)
        Contour.drawContoursToImage(self.sketch, self.contours.getShapesOfClass_x(self.canvas.parent.activeClass()))
    
    def getFinalContours(self):
        if self.sketch is None:
            return
        contours = Contour.extractContoursFromImage(self.sketch, not self.canvas.parent.allowInnerContours)

        # delete changed contours
        old_contours = self.contours.getShapesOfClass_x(self.canvas.parent.activeClass())
        changedcontours = Contour.getContoursNotinListOfContours(old_contours,contours)
        self.contours.deleteShapes(changedcontours)
        for c in contours:
            c.setClassLabel(self.canvas.parent.activeClass())
            
        self.contours.addShapes(contours)
        self.canvas.redrawImage()
        
    def finishNewContour(self, delete = False):
        if self.NewContour is not None:
            self.NewContour.closeContour()
            i = 0 if delete else 1
            cv2.drawContours(self.sketch, [self.NewContour.points], 0, (i), -1)  
        self.getFinalContours()
        self.NewContour = None
        

    def drawcontouraccessories(self, contour, painter = None):
        color = self.canvas.parent.ClassColor(contour.classlabel)
        if painter is None:
            painter = self.getPainter(color = color)
        
        if self.canvas.drawShapeNumber:
            contournumber = self.contours.getShapeNumber(contour)
            painter.drawText(self.getLabelNumPosition(contour) , str(contournumber))
            
        if self.canvas.drawSkeleton:
            skeleton = contour.getSkeleton()         
            
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