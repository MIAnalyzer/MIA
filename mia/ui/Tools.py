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

import time
import utils.shapes.Contour as Contour

class canvasTool(Enum):
    drag = 'drag'
    draw = 'draw'
    assign = 'assign'
    delete = 'delete'
    extend = 'extend'
    poly =  'poly'
    assist = 'assist'
    scale =  'scale'
    shift = 'shift'
    point = 'point'
    setimageclass = 'setimageclass'
    setallimagesclass = 'setallimagesclass'
    assignimageclass = 'assignimageclass'
    objectnumber = 'objectnumber'
    objectcolor = 'objectcolor'
    deleteobject = 'deleteobject'

class canvasToolButton(Enum):
    # Buttons are set visible or invisible depending on mode
    drag = canvasTool.drag
    draw = canvasTool.draw
    poly =  canvasTool.poly
    point = canvasTool.point
    shift = canvasTool.shift
    assign = canvasTool.assign
    extend = canvasTool.extend
    delete = canvasTool.delete
    assignimageclass = canvasTool.assignimageclass
    setimageclass = canvasTool.setimageclass
    
    

def validTool(f):
    # should decorate all functions that use canvas.painter 
    # to check if a valid tool is taken for that painter
    def wrapped_f(self, *args, **kwargs):
        if self.type in self.canvas.painter.tools:
            return f(self, *args, **kwargs)
        else:
            raise TypeError ('not a valid tool for that painter')
    return wrapped_f


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
    
    def initialize(self):
        pass

    def ShowSettings(self):
        pass

    def HideSettings(self):
        pass


class DummyTool(AbstractTool):
    # dummy tool, that executes on Button press,
    # no interaction with canvas
    def __init__(self, canvas, text):
        super().__init__(canvas)
        self.Text = text
        self.canvas.setnewTool(canvasTool.drag.name)
        
    def mouseMoveEvent(self, e):
        pass
    def mouseReleaseEvent(self,e):
        pass
    def mousePressEvent(self,e):
        pass
    def Cursor(self):
        return Qt.ArrowCursor  
    
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
        if self.canvas.hasImage():
            self.canvas.setDragMode(QGraphicsView.ScrollHandDrag)
                        

    def wheelEvent(self,e):
        if e.angleDelta().y() > 0:
            self.canvas.zoomIn()
        else:
            self.canvas.zoomOut()
                
    def Cursor(self):
        return Qt.OpenHandCursor
                
class DrawTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Draw"
        self.type = canvasTool.draw
        self.drawimage = None

    @property
    def mode(self):
        # 1 add, -1 del, 0 slice
        if self.canvas.parent.CBAddShape.isChecked():
            return 1
        elif self.canvas.parent.CBDelShape.isChecked():
            return -1
        elif self.canvas.parent.CBSliceShape.isChecked():
            return 0

    def __del__(self): 
        if self.canvas.painter.NewContour and self.canvas.painter.NewContour.numPoints() > 1 and self.drawimage is not None:
            self.canvas.copyRect(self.drawimage, self.canvas.image(),self.canvas.getActiveDrawingRect())
            self.drawimage = None            
   

    @validTool
    def mouseMoveEvent(self, e):
        if self.canvas.painter.NewContour:
            p = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
            if not self.canvas.fastPainting and self.mode != 0:
                if self.drawimage is None:
                    self.drawimage = self.canvas.image().copy()
                if self.canvas.painter.NewContour.numPoints() > 1:
                    self.canvas.copyRect(self.drawimage, self.canvas.image(),self.canvas.getActiveDrawingRect(p))
       
                p2 = self.canvas.np2QPoint(self.canvas.painter.NewContour.getLastPoint())
                self.canvas.painter.addPoint2NewContour(p)
                
                self.canvas.copyRect(self.canvas.image(),self.drawimage, self.canvas.getActiveDrawingRect())

                p3 = self.canvas.np2QPoint(self.canvas.painter.NewContour.getFirstPoint())              
                self.canvas.painter.addline(p,p3, dashed = True)
            else:
                self.canvas.painter.addPoint2NewContour(p)
           
    @validTool
    def mouseReleaseEvent(self, e):       
        if self.canvas.painter.NewContour:
            self.canvas.painter.finishNewContour(delete = self.mode < 1, close = self.mode != 0, drawaspolygon = self.mode == 0)
            self.drawimage = None
    
    @validTool            
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.canvas.painter.NewContour is None:
                self.canvas.painter.prepareNewContour()
                self.canvas.painter.NewContour = Contour.Contour(self.canvas.parent.activeClass(), self.canvas.QPoint2np(self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))))

        if e.button() == Qt.RightButton:
            if self.mode == 1:
                self.canvas.parent.CBDelShape.setChecked(True) 
            elif self.mode == 0:
                self.canvas.parent.CBAddShape.setChecked(True)
            elif self.mode == -1:
                self.canvas.parent.CBSliceShape.setChecked(True)
            self.canvas.setCursor(self.Cursor())
       
    def Cursor(self):
        return Qt.ArrowCursor

    def ShowSettings(self):
        self.canvas.parent.DrawSettings.show()

    def HideSettings(self):
        self.canvas.parent.DrawSettings.hide()
    
    
class PolygonTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Polygon"
        self.type = canvasTool.poly
        self.drawimage = None
        self.lock = threading.Lock()
        
    def __del__(self): 
        if self.canvas.painter.NewContour and self.canvas.painter.NewContour.numPoints() > 1 and self.drawimage is not None:
            self.canvas.copyRect(self.drawimage, self.canvas.image(),self.canvas.getFieldOfViewRect())
            self.drawimage = None    
        
    @property
    def mode(self):
        # 1 add, -1 del, 0 slice
        if self.canvas.parent.CBAddShape.isChecked():
            return 1
        elif self.canvas.parent.CBDelShape.isChecked():
            return -1
        elif self.canvas.parent.CBSliceShape.isChecked():
            return 0

    @validTool
    def mouseMoveEvent(self, e):
        if self.canvas.painter.NewContour is not None:
            if not self.canvas.fastPainting:
                p2 = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
                if self.drawimage is None:
                    self.drawimage = self.canvas.image().copy()
                self.canvas.copyRect(self.drawimage, self.canvas.image(),self.canvas.getFieldOfViewRect())
                p1 = self.canvas.np2QPoint(self.canvas.painter.NewContour.getLastPoint())
                self.canvas.painter.addline(p1,p2, dashed = True)              
    
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.RightButton:
            if self.canvas.painter.NewContour:
                self.canvas.painter.finishNewContour(delete = self.mode < 1, close = self.mode != 0, drawaspolygon = self.mode == 0)
                self.drawimage = None
            else:
                if self.mode == 1:
                    self.canvas.parent.CBDelShape.setChecked(True) 
                elif self.mode == 0:
                    self.canvas.parent.CBAddShape.setChecked(True)
                elif self.mode == -1:
                    self.canvas.parent.CBSliceShape.setChecked(True)
                self.canvas.setCursor(self.Cursor())

        elif e.button() == Qt.LeftButton:
            if self.canvas.painter.NewContour is None:
                self.canvas.painter.prepareNewContour()
                self.canvas.painter.NewContour = Contour.Contour(self.canvas.parent.activeClass(), self.canvas.QPoint2np(self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))))
            else:
                p = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
                self.canvas.painter.addPoint2NewContour(p)
                if self.drawimage is None:
                    self.drawimage = self.canvas.image().copy()
                self.canvas.copyRect(self.canvas.image(),self.drawimage, self.canvas.getFieldOfViewRect())
     
    @validTool                 
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor

    def ShowSettings(self):
        self.canvas.parent.DrawSettings.show()

    def HideSettings(self):
        self.canvas.parent.DrawSettings.hide()
    
class AssignTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Assign"
        self.type = canvasTool.assign

    def mouseMoveEvent(self, e):
        pass
        
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            contour = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())),self.canvas.FontSize)
            if contour is not None:
                contour.setClassLabel(self.canvas.parent.activeClass())
                self.canvas.painter.checkForChanges()
                
                
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
        
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            
            contour = self.canvas.painter.shapes.getShapeOfClass_x(self.canvas.parent.activeClass(),self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())), self.canvas.FontSize)
            if contour is None:
                contour = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())), self.canvas.FontSize)
            if contour is not None:
                self.canvas.painter.shapes.deleteShape(contour)
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
        self.type = canvasTool.extend
        self.inprogress = False
        self.lock = threading.RLock()
        
    @validTool
    def mouseMoveEvent(self, e):
        if self.canvas.painter.sketch is None:
            return
        with self.lock:
            # if e.button() == Qt.LeftButton: is not working here for some reasons
            if self.inprogress:
                p0 = self.canvas.mapToScene(e.pos())
                p = self.canvas.QPoint2np(p0)
                x = p[0,0]
                y = p[0,1]
                # self.canvas.painter.addCircle(p0, self.canvas.parent.SSize.value(), erase = self.canvas.parent.CBErase.isChecked())
                self.canvas.painter.addCircle(p0, self.canvas.parent.SSize.value())
                val = 1 if self.canvas.parent.CBErase.isChecked() == False else 0
                cv2.circle(self.canvas.painter.sketch, (x, y), self.canvas.parent.SSize.value(), (val), -1)

    @validTool 
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            self.canvas.painter.finishNewContour()
            self.inprogress = False

    @validTool                 
    def mousePressEvent(self,e):
        if self.canvas.image() is None:
            return
        if e.button() == Qt.LeftButton:  
            self.size = self.canvas.parent.SSize.value()
            self.canvas.painter.prepareNewContour()
            self.inprogress = True
       
        elif e.button() == Qt.RightButton:
            self.canvas.parent.CBExtend.setChecked(True) if self.canvas.parent.CBErase.isChecked() else self.canvas.parent.CBErase.setChecked(True)
            self.canvas.setCursor(self.Cursor())
    
    @validTool
    def wheelEvent(self,e):
        if e.angleDelta().y() > 0:
            v = self.canvas.parent.SSize.value() + 1
            self.canvas.parent.SSize.setValue(v)
        else:
            v = self.canvas.parent.SSize.value() - 1
            self.canvas.parent.SSize.setValue(v)
        self.size = self.canvas.parent.SSize.value()
        self.canvas.setCursor(self.Cursor())

    def createCursor(self):
        if not self.canvas.hasImage() or not self.canvas.zoomfactor():
            return Qt.ArrowCursor
        
        size = (2*self.canvas.parent.SSize.value()) * self.canvas.zoomfactor()[0]
        # unfortunately max size is somewhere around 100
        if size > 95:
            return Qt.SizeAllCursor
        icon = QPixmap(size+1,size+1)
        icon.fill(Qt.transparent)
        p = QPainter(icon)
        color = self.canvas.parent.ClassColor()
        if self.canvas.parent.CBErase.isChecked():
            p.setBrush(QBrush(Qt.NoBrush))
            p.setPen(QPen(color, Qt.SolidLine))
        else:
            p.setBrush(QBrush(color,Qt.SolidPattern))
            p.setPen(QPen(Qt.NoPen))
            
        p.drawEllipse(0, 0, size, size)
        p.end()
        return QCursor(icon, -1, -1)

    def Cursor(self):
        return self.createCursor()
    
    def ShowSettings(self):
        self.canvas.parent.ExtendSettings.show()

    def HideSettings(self):
        self.canvas.parent.ExtendSettings.hide()


class DEXTRTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = 'Assisted Segmentation'
        self.type = canvasTool.assist
        self.extremePoints = []

    def __del__(self): 
        self.extremePoints.clear()
        self.canvas.redrawImage()
        
    def mouseMoveEvent(self, e):
        pass

    @validTool  
    def mouseReleaseEvent(self,e):
        if self.canvas.image() is None:
            return
        if e.button() == Qt.LeftButton:  
            p = self.canvas.mapToScene(e.pos())

            h = self.canvas.pen_size
            w = self.canvas.pen_size
            x = p.x() - w//2
            y = p.y() - h//2

            self.canvas.painter.addRectangle(x, y, h, w)
            self.extremePoints.append(self.canvas.QPoint2npPoint(self.canvas.f2intPoint(p)))
            if len(self.extremePoints) == 4:
                self.canvas.painter.assistedSegmentation(self.extremePoints)
                self.extremePoints.clear()

    def mousePressEvent(self,e):
        pass
    def Cursor(self):
        return Qt.ArrowCursor  

        
class PointTool(AbstractTool):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Point"
        self.type = canvasTool.point
        
    def mouseMoveEvent(self, e):
        pass

     
    @validTool   
    def mouseReleaseEvent(self,e):
        if self.canvas.image() is None:
            return
        if e.button() == Qt.LeftButton:
            p = self.canvas.f2intPoint(self.canvas.mapToScene(e.pos()))
            self.canvas.painter.addPoint(p)
            
        elif e.button() == Qt.RightButton:
            self.canvas.setnewTool(canvasTool.delete.name)

          
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.CrossCursor      
    
class ShiftTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Shift"
        self.type = canvasTool.shift
        self.shiftedShape = None
        self.shifting = False
        self.origin_coordinates = None

    @validTool    
    def mouseMoveEvent(self, e):
        if self.shiftedShape and self.shifting:
            p = self.canvas.mapToScene(e.pos())
            self.shiftedShape.coordinates = self.canvas.QPoint2npPoint(self.canvas.f2intPoint(p))
            if not self.canvas.fastPainting:
                self.canvas.redrawImage(True)

    @validTool    
    def mouseReleaseEvent(self,e):
        if not self.shiftedShape:
            return
        self.shifting = False

        self.shiftedShape = None
        self.canvas.redrawImage()

    @validTool     
    def mousePressEvent(self,e):
        if self.canvas.image() is None:
            return
        p = self.canvas.mapToScene(e.pos())

        shape = self.canvas.painter.shapes.getShapeOfClass_x(self.canvas.parent.activeClass(),self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())), self.canvas.FontSize)
        if shape is None:
            shape = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())), self.canvas.FontSize)
        if shape is not None:
            self.shifting = True
            self.shiftedShape = shape
            self.origin_coordinates = shape.coordinates


    def Cursor(self):
        return Qt.SizeAllCursor   


class ScaleTool(AbstractTool):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Scale"
        self.type = canvasTool.scale
        self.size = 15
        self.color = QColor(0,0,255)
        self.num = 0
        self.p1 = None
        self.dist_inpixel = 0
        
    def mouseMoveEvent(self, e):
        pass

        
    def mouseReleaseEvent(self,e):
        if self.canvas.image() is None:
            return
        if e.button() == Qt.LeftButton:
            if self.num == 0:
                self.p1 = self.canvas.mapToScene(e.pos())
                self.canvas.painter.addTriangle(self.p1, self.size, self.color)
                self.num += 1
            else:  
                p2 = self.canvas.mapToScene(e.pos())
                self.canvas.painter.addTriangle(p2, self.size, self.color)
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
    

# tracking tools
class ObjectNumberTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "ObjectNumber"
        self.type = canvasTool.objectnumber

    def mouseMoveEvent(self, e):
        pass
        
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            shape = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())),self.canvas.FontSize)
            if shape is not None:
                objnum,ok = QInputDialog.getInt(self.canvas.parent, "Set Object Number","Object Number", shape.objectNumber, 1, 100000)
                if ok and objnum > 0:
                    old_obj = self.canvas.painter.shapes.getShapeNumberX(objnum)
                    if old_obj:
                        old_obj.objectNumber = self.canvas.parent.tracking.getFirstFreeObjectNumber()
                    oldobjnum = self.canvas.parent.tracking.setObjectNumber(obj=shape, objnum=objnum)
                    self.canvas.redrawImage()
                     
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor

class ObjectColorTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "ObjectColor"
        self.type = canvasTool.objectcolor

    def mouseMoveEvent(self, e):
        pass
        
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            shape = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())),self.canvas.FontSize)
            if shape is not None:
                color = QColorDialog.getColor()
                if color.isValid():
                    self.canvas.painter.objectcolors[shape.objectNumber-1] = color
                    self.canvas.redrawImage()

    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor

class DeleteObjectTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "DeleteObject"
        self.type = canvasTool.deleteobject

    def mouseMoveEvent(self, e):
        pass
        
    @validTool
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            shape = self.canvas.painter.shapes.getShape(self.canvas.QPoint2np(self.canvas.mapToScene(e.pos())),self.canvas.FontSize)
            if shape is not None:
                with self.canvas.parent.wait_cursor():
                    self.canvas.parent.tracking.deleteObject(shape)
                    self.canvas.painter.shapes.deleteShape(shape)
                    self.canvas.redrawImage()
  
    def mousePressEvent(self,e):
        pass

    def Cursor(self):
        return Qt.ArrowCursor
    

# classification tools
class setAllImagesClassTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Set all images to class"
        self.type = canvasTool.setallimagesclass
        
    def initialize(self):
        # a bit hacky and poor designed as this is the execute
        for i in range(self.canvas.parent.files.numofImages):  
            self.canvas.assignImageClass()
            self.canvas.parent.nextImage()
        self.canvas.setnewTool(canvasTool.drag.name)

    def mouseReleaseEvent(self,e):
        pass
    def mouseMoveEvent(self,e):
        pass
    def mousePressEvent(self,e):
        pass
    def Cursor(self):
        return Qt.ArrowCursor  
    
class setImageClassTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Set image class"
        self.type = canvasTool.setimageclass
        
    def initialize(self):
        # a bit hacky and poor designed as this is the execute  
        self.canvas.assignImageClass()
        self.canvas.parent.nextImage()
        self.canvas.setnewTool(canvasTool.drag.name)

    def mouseReleaseEvent(self,e):
        pass
    def mouseMoveEvent(self,e):
        pass
    def mousePressEvent(self,e):
        pass
    def Cursor(self):
        return Qt.ArrowCursor  
        
        
class assignImageClassTool(AbstractTool):
    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.Text = "Assign image class"
        self.type = canvasTool.assignimageclass
        
    def initialize(self):
        self.canvas.assignImageClass()
        self.canvas.setnewTool(canvasTool.drag.name)
        
    def mouseReleaseEvent(self,e):
        pass
    def mouseMoveEvent(self,e):
        pass
    def mousePressEvent(self,e):
        pass
    def Cursor(self):
        return Qt.ArrowCursor  