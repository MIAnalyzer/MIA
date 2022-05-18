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
from utils.shapes.Point import Points, Point

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from ui.painter.ContourPainter import ContourPainter
from ui.painter.ObjectPainter import ObjectPainter
from ui.painter.ImageLabelPainter import ImageLabelPainter
from ui.painter.NoPainter import NoPainter

from dl.method.mode import dlMode


# to do: derive QGraphicsItem to display whole slide image
# get the viewpoint via "self.mapToScene(self.viewport().geometry()).boundingRect()"
# and paint target region

class DrawingImage(QGraphicsItem):
    # need to inherit here because the pixmap of QGraphicsPixmapItem is const
    # like this we can draw directly on the pixmap, avoid unnecessary copies
    def __init__(self, pixmap, parent=None):
        super(DrawingImage, self).__init__(parent)
        self.BackgroundImage = pixmap
        self.invisibleCanvas = QPixmap()
        
    def setImage(self, pixmap):
        if pixmap:
            self.BackgroundImage = pixmap
            self.invisibleCanvas = QPixmap(self.BackgroundImage.size())
            self.invisibleCanvas.fill(QColor.fromRgb(0,0,0,0))

            
    def setCanvas(self, pixmap):
        self.invisibleCanvas = pixmap
        
    
    def getCanvas(self):
        if self.hasImage():
            return self.invisibleCanvas
        else:
            return None
        
    def getImage(self):
        if self.hasImage():
            return self.BackgroundImage
        else:
            return None
        
    def hasImage(self):
        return self.BackgroundImage.width()*self.BackgroundImage.height() != 0    
    
            
    def reset(self):
        self.invisibleCanvas.fill(QColor.fromRgb(0,0,0,0))
        
    def boundingRect(self):
        if self.BackgroundImage is None:
            return QRectF()
        return QRectF(0, 0, self.BackgroundImage.width(), self.BackgroundImage.height())
    
    def paint(self, painter, option, widget):
        painter.drawPixmap(0, 0, self.BackgroundImage)
        painter.drawPixmap(0, 0, self.invisibleCanvas)



class Canvas(QGraphicsView):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.colors = []
        
        self.bgcolor = QColor(0,0,0)
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        
        # drawing contour types
        self.painter = NoPainter(self)

        self.enableToggleTools = True
        self.FontSize = 18
        self.pen_size = 3
        self.ContourTransparency = 50
        self.drawShapeNumber = False
        self.drawSkeleton = False
        self.skeletonsmoothingfactor = 11
        self.minContourSize = 3
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
        return self.displayedimage.getCanvas()
              
    def updateImage(self):
        if self.image() is not None:
            self.update()
       
    def checkForChanges(self):
        self.painter.checkForChanges()
        
    def resetView(self, scale=True):
        if not self.hasImage():
            return
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

    def zoomIn(self):
        factor = 1.25
        self.zoomstep += 1
        if self.zoomstep > 0:
            self.scale(factor, factor)

    def zoomOut(self):
        factor = 0.8
        self.zoomstep -= 1
        if self.zoomstep > 0:
            self.scale(factor, factor)
        elif self.zoomstep == 0:
            self.resetView()
        else:
            self.zoomstep = 0

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

    def reset(self):
        self.displayedimage.setImage(QPixmap())

    def ReloadImage(self, resetView = True):
        if self.parent.files.CurrentImagePath() is None:
            return
        self.displayedimage.setImage(self.getCurrentPixmap())
        self.ReloadLabels(resetView)
        self.painter.clearTracking()

            
    def ReloadLabels(self, resetView = True):
        if not self.image():
            return
        self.clearLabel()
        self.getLabel()
        self.redrawImage()

        if resetView:
            self.resetView()

    def undoDrawing(self):
        if self.painter.undoPossible():
            self.painter.undo()
            self.redrawImage()
                    
    def getCurrentPixmap(self):
        image = self.parent.getCurrentImage() 
        if image is None:
            self.parent.PopupWarning('No image loaded')
            return
        return self.convert2Pixmap(image)
        
    def convert2Pixmap(self, image):
        image = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,QImage.Format_RGB888).rgbSwapped() 
        return QPixmap(image)
        
    def hasImage(self):
        return self.displayedimage.hasImage()
    
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
        self.displayedimage.setImage(pix)
        
    def SaveCurrentLabel(self):
        self.painter.save()
            
    def getLabel(self):
        self.painter.load()
    
    def redrawImage(self, quickdraw=False):
        if self.hasImage():
            self.displayedimage.reset()
            self.drawLabel(not quickdraw)

            
    def copyRect(self, source, target, rec):
        p = QPainter(target)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(rec, source , rec)     
        
    def getActiveDrawingRect(self, newpoint=None):
        x,y,w,h = self.painter.NewContour.getBoundingBox()
        i = self.pen_size
        if newpoint:
            if w*h == 0:
                bbox = QRect(min(x-i,newpoint.x()-i),min(y-i,newpoint.y()-i),abs(newpoint.x()-x)+2*i,abs(newpoint.y()-y)+2*i)
            else:
                bbox = QRect(min(x-i,newpoint.x()),min(y-i,newpoint.y()),max(w,newpoint.x()-x)+2*i,max(h+2*i,newpoint.y()-y)+2*i)
        else:
            bbox = QRect(x-i,y-i,w+i,h+i)
        fov = self.getFieldOfViewRect()
        return bbox.intersected(fov.toRect())
            
    def drawLabel(self, shapeschanged=True):
        if self.image() is None:
            return  
        if shapeschanged:
            self.parent.ShapesChanged()
        self.painter.draw()
        self.updateImage()
        
        
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
        elif tool == canvasTool.draw.name:
            self.tool = Tools.DrawTool(self)
        elif tool == canvasTool.assign.name:
            self.tool = Tools.AssignTool(self)
        elif tool == canvasTool.extend.name:
            self.tool = Tools.ExtendTool(self)
        elif tool == canvasTool.delete.name:
            self.tool = Tools.DeleteTool(self)
        elif tool == canvasTool.poly.name:
            self.tool = Tools.PolygonTool(self)
        elif tool == canvasTool.assist.name:
            self.tool = Tools.DEXTRTool(self)
        elif tool == canvasTool.scale.name:
            self.tool = Tools.ScaleTool(self)
        elif tool == canvasTool.point.name:
            self.tool = Tools.PointTool(self)
        elif tool == canvasTool.shift.name:
            self.tool = Tools.ShiftTool(self)
        elif tool == canvasTool.setimageclass.name:
            self.tool = Tools.setImageClassTool(self)
        elif tool == canvasTool.setallimagesclass.name:
            self.tool = Tools.setAllImagesClassTool(self)
        elif tool == canvasTool.assignimageclass.name:
            self.tool = Tools.assignImageClassTool(self)
        elif tool == canvasTool.objectnumber.name:
            self.tool = Tools.ObjectNumberTool(self)
        elif tool == canvasTool.objectcolor.name:
            self.tool = Tools.ObjectColorTool(self)
        elif tool == canvasTool.deleteobject.name:
            self.tool = Tools.DeleteObjectTool(self)

        if self.tool.type not in self.painter.tools:
            self.tool = Tools.DragTool(self)



        self.tool.initialize()
        self.tool.ShowSettings()
        self.parent.writeStatus('Current Tool: ' + self.tool.Text) 
        self.updateCursor()
        
    def setCanvasMode(self, mode):
        self.enableToggleTools = True
        if mode == dlMode.Segmentation:
            self.painter = ContourPainter(self)
            self.painter.smartmode = self.parent.CBSmartMode.isChecked()
            self.setMinimumContourSize(self.minContourSize)
        elif mode == dlMode.Object_Counting:
            self.painter = ObjectPainter(self)
        elif mode == dlMode.Classification:
            self.painter = ImageLabelPainter(self)
            self.enableToggleTools = False
        else:
            self.painter = NoPainter(self)
            self.enableToggleTools = False
        self.parent.setToolButtons()
        
    def updateCursor(self):
        self.setCursor(self.tool.Cursor())

    def toggleDrag(self):
        if not self.enableToggleTools:
            return
            
        if self.tool.type == canvasTool.drag:
            if self.lasttool.type in self.painter.tools:
                self.setnewTool(self.lasttool.type.name)
        else:
            self.setnewTool(canvasTool.drag.name)
            
            
    def setMinimumContourSize(self, size):
        self.minContourSize = size
        if isinstance(self.painter, ContourPainter):
            self.painter.shapes.minSize = self.minContourSize

    def getFieldOfViewImage(self, image=None):
        if image is None:
            image = self.parent.getCurrentImage() 
        fov = self.getFieldOfViewRect()
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
