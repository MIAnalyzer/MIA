# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:44:53 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os
import numpy as np
import cv2

import ui.Tools as Tools
from ui.Tools import canvasTool
from utils.Point import Points, Point

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from ui.ui_ContourPainter import ContourPainter
from ui.ui_ObjectPainter import ObjectPainter
from ui.ui_ImageLabelPainter import ImageLabelPainter
from ui.ui_NoPainter import NoPainter

from dl.DeepLearning import dlMode

import time

# to do: derive QGraphicsItem to display whole slide image
# get the viewpoint via "self.mapToScene(self.viewport().geometry()).boundingRect()"
# and paint target region

class DrawingImage(QGraphicsItem):
    # need to inherit here because the pixmap of QGraphicsPixmapItem is const
    # like this we can draw directly on the pixmap, avoid unnecessary copies
    def __init__(self, pixmap, parent=None):
        super(DrawingImage, self).__init__(parent)
        self.pmap = pixmap
        
    def setPixmap(self, pixmap):
        if pixmap:
            self.pmap = pixmap
   
    def pixmap(self):
        if self.pmap == QPixmap():
            return None
        else:
            return self.pmap
        
    def boundingRect(self):
        if self.pmap is None:
            return QRectF()
        return QRectF(0, 0, self.pmap.width(), self.pmap.height())
    
    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.pmap)


class Canvas(QGraphicsView):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.colors = []
        
        self.bgcolor = QColor(0,0,0)
        self.rawimage = None
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        
        # drawing contour types
        self.painter = NoPainter(self)

        self.enableToggleTools = True
        self.FontSize = 18
        self.pen_size = 3
        self.ContourTransparency = 50
        self.drawShapeNumber = True
        self.drawSkeleton = False
        self.minContourSize = 100
        self.fastPainting = False
        
        self.displayedimage = DrawingImage(QPixmap())
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.displayedimage)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)
        
        self.scale_pixel_per_mm = 1

    def mouseMoveEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mouseMoveEvent(e)
        super(Canvas, self).mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mouseReleaseEvent(e)
        super(Canvas, self).mouseReleaseEvent(e)

    def mousePressEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mousePressEvent(e)
        super(Canvas, self).mousePressEvent(e)                   

    def wheelEvent(self,e):
        if self.hasImage():
            self.tool.wheelEvent(e)
            
    def image(self):
        return self.displayedimage.pixmap()
            
    def updateImage(self):
        if self.image() is not None:
            self.update()
       
    def checkForChanges(self):
        self.painter.checkForChanges()
        
    def resetView(self, scale=True):
        rect = QRectF(self.image().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                if scenerect.width() * scenerect.height() == 0:
                    return
                factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                self.zoomstep = 0
                self.parent.updateCursor()

    def zoomfactor(self):
        if not self.hasImage():
            return None
        rect = QRectF(self.image().rect()) 
        viewrect = self.viewport().rect()
        scenerect = self.transform().mapRect(rect)
        if rect.width() * rect.height() == 0:
            return None
        w = scenerect.width() / rect.width()
        h = scenerect.height() / rect.height()
        return (w,h)

    def reset(self, width, height):
        pix = QPixmap(width, height).fill(self.bgcolor)
        self.displayedimage.setPixmap(pix)
        
    def ReloadImage(self, resetView = True):
        if self.parent.files.CurrentImagePath() is None:
            return
        self.rawimage = self.getCurrentPixmap()
        self.clearLabel()
        self.getLabel()
        self.redrawImage()
        if resetView:
            self.resetView()
                    
    def getCurrentPixmap(self):
        image = self.parent.getCurrentImage() 
        if image is None:
            self.parent.PopupWarning('No image loaded')
            return
        image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,QImage.Format_RGB888).rgbSwapped() 
        return QPixmap(image)
        
    def hasImage(self):
        return self.image() is not None
    
    def setScale(self,scale_ppmm):
        if scale_ppmm <= 0:
            return
        self.scale_pixel_per_mm = scale_ppmm
        if not self.parent.results_form.LEScale.text() or int(self.scale_pixel_per_mm+0.5) != int(self.parent.results_form.LEScale.text()):
            self.parent.results_form.LEScale.setText(str(int(self.scale_pixel_per_mm+0.5)))
        
    def zoom(self):
        width = self.geometry().width()
        height = self.geometry().height()
        factor = 1 + self.zoomstep/10 
        pix = self.image().scaled(width * factor, height * factor, Qt.KeepAspectRatio)
        self.displayedimage.setPixmap(pix)
        
    def SaveCurrentLabel(self):
        self.painter.save()
            
    def getLabel(self):
        self.painter.load()
    
    def redrawImage(self):
        if self.rawimage is not None:
            self.displayedimage.setPixmap(self.rawimage.copy())
            self.drawLabel()
            
    def drawLabel(self):
        if self.image() is None:
            return  
        self.painter.draw()

        self.updateImage()
        self.parent.numOfShapesChanged()
        
    def clearLabel(self):
        self.painter.clear()
        
    def f2intPoint(self, p):
        # we do not round here!
        # by using floor it is ensured that the mouse arrow head is inside the selected pixel
        p.setX(int(p.x()))
        p.setY(int(p.y()))
        return p
    
    def QPoint2np(self,p):
        return np.array([(p.x(),p.y())], dtype=np.int32)
    
    def np2QPoint(self,p):
        return QPointF(p[0],p[1])

    def Point2QPoint(self,p):
        return QPointF(p.coordinates[0], p.coordinates[1])
    
    def QPoint2npPoint(self,p):
        return np.array([p.x(),p.y()])
    
    
    def setnewTool(self, tool):
        self.lasttool = self.tool  
        self.updateImage()
        self.tool.HideSettings()
        if tool == canvasTool.drag.name:
            self.tool = Tools.DragTool(self)
        if tool == canvasTool.draw.name:
            self.tool = Tools.DrawTool(self)
        if tool == canvasTool.assign.name:
            self.tool = Tools.AssignTool(self)
        if tool == canvasTool.extend.name:
            self.tool = Tools.ExtendTool(self)
        if tool == canvasTool.delete.name:
            self.tool = Tools.DeleteTool(self)
        if tool == canvasTool.poly.name:
            self.tool = Tools.PolygonTool(self)
        if tool == canvasTool.scale.name:
            self.tool = Tools.ScaleTool(self)
        if tool == canvasTool.point.name:
            self.tool = Tools.PointTool(self)
        if tool == canvasTool.shift.name:
            self.tool = Tools.ShiftTool(self)
        
        self.tool.ShowSettings()
        self.parent.writeStatus('Current Tool: ' + self.tool.Text) 
        self.updateCursor()
        
    def setCanvasMode(self, mode):

        if mode == dlMode.Segmentation:
            self.painter = ContourPainter(self)
            self.setMinimumContourSize(self.minContourSize)
        elif mode == dlMode.Object_Counting:
            self.painter = ObjectPainter(self)
        elif mode == dlMode.Classification:
            self.painter = ImageLabelPainter(self)
        else:
            self.painter = NoPainter(self)
        self.parent.setToolButtons()
        
    def updateCursor(self):
        self.setCursor(self.tool.Cursor())

    def toggleDrag(self):
        if not self.enableToggleTools:
            return
            
        if self.tool.type == canvasTool.drag:
            self.setnewTool(self.lasttool.type.name)
        else:
            self.setnewTool(canvasTool.drag.name)
            
            
    def setMinimumContourSize(self, size):
        self.minContourSize = size
        if isinstance(self.painter, ContourPainter):
            self.painter.shapes.minSize = self.minContourSize

    def getFieldOfViewImage(self):
        fov = self.getFieldOfViewRect()
        image = self.parent.getCurrentImage() 
        y0 = int(fov.x())
        y1 = int(fov.x() + fov.width())
        x0 = int(fov.y())
        x1 = int(fov.y() + fov.height())
        image = image[x0:x1,y0:y1]
        return image
        
    def getFieldOfViewRect(self):
        fov = self.mapToScene(self.viewport().geometry()).boundingRect()
        fov.setX(max(0,fov.x()))
        fov.setY(max(0,fov.y()))
        return fov
    
    def assignImageClass(self):
        assert isinstance(self.painter,ImageLabelPainter) 
        self.painter.addImageLabel()
