# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:38:48 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from abc import ABC, abstractmethod
from enum import Enum
import cv2
import numpy as np
import threading


import utils.Contour as Contour

class canvasTool(Enum):
    drag = 'drag'
    draw = 'draw'
    assign = 'assign'
    delete = 'delete'
    expand = 'extend'
    poly =  'poly'
    scale =  'scale'


class AbstractTool(ABC):
    def __init__(self, canvas):
        self.canvas = canvas
        self.Text = None
        self.type = None
        super().__init__()
        
    @abstractmethod
    def mouseMoveEvent(self, e):
        pass
    @abstractmethod
    def mouseReleaseEvent(self,e):
        pass
    @abstractmethod
    def mousePressEvent(self,e):
        pass

    def wheelEvent(self,e):
        pass

    @abstractmethod
    def Cursor(self):
        pass


    
    
class DragTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Drag"
        self.type = canvasTool.drag
    
    def mouseMoveEvent(self, e):
        pass


    def mouseReleaseEvent(self, e):
        if self.canvas.dragMode() == QGraphicsView.ScrollHandDrag:
            self.canvas.setDragMode(QGraphicsView.NoDrag)
            self.canvas.setCursor(Qt.OpenHandCursor)
            
    def mousePressEvent(self, e):
        if not self.canvas.displayedimage.pixmap().isNull():
            self.canvas.setDragMode(QGraphicsView.ScrollHandDrag)
                        

    def wheelEvent(self,e):
        if e.angleDelta().y() > 0:
            factor = 1.25
            self.canvas.zoomstep += 1
        else:
            factor = 0.8
            self.canvas.zoomstep -= 1
        if self.canvas.zoomstep > 0:
            self.canvas.scale(factor, factor)
        elif self.canvas.zoomstep == 0:
            self.canvas.fitInView()
        else:
            self.canvas.zoomstep = 0
                
    def Cursor(self):
        return Qt.OpenHandCursor
                
class DrawTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Draw"
        self.type = canvasTool.draw
   
    def mouseMoveEvent(self, e):
        if self.canvas.activeContour:
            # order imaportant
            p1 = self.canvas.activeContour.getLastPoint()
            p2 = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
            if (self.canvas.addPoint2Contour(p2)):
                self.canvas.addline(p1,p2)

    def mouseReleaseEvent(self, e):       
        if self.canvas.activeContour:
            p1 = self.canvas.activeContour.getFirstPoint()
            p2 = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
            self.canvas.addPoint2Contour(p2)
            if (self.canvas.activeContour.isValid()):
                self.canvas.addline(p1,p2)
            self.canvas.finishContour()
                
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.canvas.activeContour is None:
                self.canvas.activeContour = Contour.Contour(self.canvas.parent.activeClass(), self.canvas.f2intPoint(self.canvas.mapToScene(e.pos())))
       
    def Cursor(self):
        return Qt.ArrowCursor
    
    
class PolygonTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Polygon"
        self.type = canvasTool.poly
        
    def mouseMoveEvent(self, e):
        pass
        
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.RightButton:
            if self.canvas.activeContour:
                p1 = self.canvas.activeContour.getLastPoint()
                p2 = self.canvas.activeContour.getFirstPoint()
                self.canvas.addPoint2Contour(p2)
                self.canvas.addline(p1,p2)
                self.canvas.finishContour()
        elif e.button() == Qt.LeftButton:
            if self.canvas.activeContour is None:
                self.image = self.canvas.image.copy()
                self.canvas.activeContour = Contour.Contour(self.canvas.parent.activeClass(), self.canvas.f2intPoint(self.canvas.mapToScene(e.pos())))
            else:
                p1 = self.canvas.activeContour.getLastPoint()
                p2 = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
                if (self.canvas.addPoint2Contour(p2)):
                    self.canvas.addline(p1,p2)
                      
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor
    
class AssignTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Assign"
        self.type = canvasTool.assign

        
    def mouseMoveEvent(self, e):
        pass
        
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            contour = self.canvas.Contours.getContour(self.canvas.mapToScene(e.pos()))
            if contour is not None:
                contour.classlabel = self.canvas.parent.activeClass()
                self.canvas.redrawImage()
                      
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor
        
class DeleteTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Delete"
        self.type = canvasTool.delete
        
        
    def mouseMoveEvent(self, e):
        pass
        
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            contour = self.canvas.Contours.getContour(self.canvas.mapToScene(e.pos()))
            if contour is not None:
                self.canvas.Contours.deleteContour(contour)
                self.canvas.redrawImage()
                      
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor
        
    
class ExtendTool(AbstractTool):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Extend/Erase"
        self.type = canvasTool.expand
        self.size = 10
        self.sketch = None
        self.contours = None
        self.lock = threading.RLock()
        self.erase = False
        
    def mouseMoveEvent(self, e):
        if self.sketch is None:
            return
        with self.lock:
            p0 = self.canvas.mapToScene(e.pos())
            p = Contour.QPoint2np(p0)
            
            x = p[0,0]
            y = p[0,1]
            self.canvas.addcircle(p0, self.size)
            val = 1 if self.erase == False else 0
            cv2.circle(self.sketch, (x, y), self.size, (val), -1)

        
    def mouseReleaseEvent(self,e):
        if self.sketch is None:
            return

        contours = Contour.extractContoursFromImage(self.sketch)

        # returns all contours in self.contours not present in countours
        # fastest way I found so far
        c1 = [(x, cv2.moments(x.points),cv2.boundingRect(x.points) ) for x in self.contours]
        c2 = [(cv2.moments(x),cv2.boundingRect(x) ) for x in contours]   
        changedcontours = [x[0] for x in c1 if x[1:] not in c2]
     
        self.canvas.Contours.deleteContours(changedcontours)
        self.canvas.Contours.addContours([Contour.Contour(self.canvas.parent.activeClass(),c) for c in contours])

        self.canvas.redrawImage()
        self.sketch = None
        self.contours = None
                      
    def mousePressEvent(self,e):
        if self.canvas.image is None:
            return
        if e.button() == Qt.LeftButton:
            
            self.size = self.canvas.parent.SSize.value()
            self.erase = self.canvas.parent.CBErase.isChecked()
            
            self.contours = self.canvas.Contours.getContoursOfClass_x(self.canvas.parent.activeClass())
            width = self.canvas.image.width()
            height = self.canvas.image.height()
            self.sketch = np.zeros((height, width, 1), np.uint8)
            Contour.drawContoursToImage(self.sketch, self.contours)
            
        elif e.button() == Qt.RightButton:
            self.canvas.parent.CBExtend.setChecked(True) if self.erase else self.canvas.parent.CBErase.setChecked(True)
            self.erase = self.canvas.parent.CBErase.isChecked()
    
    def wheelEvent(self,e):
        if e.angleDelta().y() > 0:
            v = self.canvas.parent.SSize.value() + 1
            self.canvas.parent.SSize.setValue(v)
        else:
            v = self.canvas.parent.SSize.value() - 1
            self.canvas.parent.SSize.setValue(v)

    def Cursor(self):
        return Qt.ArrowCursor
    

class ScaleTool(AbstractTool):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Scale"
        self.type = canvasTool.scale
        self.size = 15
        self.color = Qt.blue
        self.num = 0
        self.p1 = None
        self.dist_inpixel = 0
        
    def mouseMoveEvent(self, e):
        pass

        
    def mouseReleaseEvent(self,e):
        if self.canvas.image is None:
            return
        if e.button() == Qt.LeftButton:
            if self.num == 0:
                self.p1 = self.canvas.mapToScene(e.pos())
                self.canvas.addTriangle(self.p1, self.size, self.color)
                self.num += 1
            else:  
                p2 = self.canvas.mapToScene(e.pos())
                self.canvas.addTriangle(p2, self.size, self.color)
                dist_inpixel = abs(self.p1.x() - p2.x())
                dist_inum,ok = QInputDialog.getInt(self.canvas.parent, "Set Scale","enter distance in \u03BCm")
                if ok and dist_inum > 0:
                   self.canvas.setScale(dist_inpixel / dist_inum * 1000)
                
                self.canvas.setnewTool(canvasTool.drag.name)
                self.canvas.redrawImage()
                self.num = 0
                
        
                      
    def mousePressEvent(self,e):
        pass
    
    

    def Cursor(self):
        return Qt.UpArrowCursor