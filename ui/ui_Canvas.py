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
import utils.Contour as Contour
from utils.Point import Points, Point

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from ui.ui_ContourPainter import ContourPainter
from ui.ui_ObjectPainter import ObjectPainter
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
        self.pen_size = 3
        self.bgcolor = QColor(0,0,0)
#        self.image = None
        self.rawimage = None
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        
        # drawing contour types
        self.painter = NoPainter(self)
        # self.Contours = Contour.Contours()
        # self.Points = Points()
           
        # self.NewContour = None
        self.enableToggleTools = True
        self.FontSize = 18
        self.ContourTransparency = 50
        self.drawContourNumber = True
        self.drawSkeleton = False
        self.minContourSize = 100
        # self.sketch = None
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
      
    # def addline(self, p1 ,p2, color = None, dashed = False):
    #     p = self.getPainter(color)
    #     if dashed:
    #         pen = p.pen()
    #         pen.setStyle(Qt.DashDotDotLine)
    #         p.setPen(pen)
    #     else:
    #         pen = p.pen()
    #         pen.setStyle(Qt.SolidLine)
    #         p.setPen(pen)
        
    #     p.drawLine(p2, p1)
    #     self.updateImage()
        
    # def addTriangle(self, point, size, color = None):
    #     psize = self.pen_size
    #     self.pen_size = 1
    #     p = self.getPainter(color)
    #     polygon = QPolygonF()
    #     polygon.append(point)
    #     point2 = QPointF(point.x() + size/2, point.y() - size)
    #     polygon.append(point2)
    #     point3 = QPointF(point.x() - size/2, point.y() - size)
    #     polygon.append(point3)
    #     p.drawPolygon(polygon)
    #     self.pen_size = psize
    #     self.updateImage()
        
    # def addCircle(self, center , radius, color = None):
    #     p = self.getPainter(color)
    #     p.drawEllipse(center, radius, radius)
    #     self.updateImage()
        
    # def addPoint(self, point):
    #     self.Points.addShape(Point(self.parent.activeClass(), (point.x(), point.y())))
    #     self.redrawImage()
    
    # def deletePoint(self, position):
    #     p = self.Points.getPoint((position.x(), position.y()), 10)
    #     self.Points.deleteShape(p)
    #     self.redrawImage()
        
    def updateImage(self):
        if self.image() is not None:
            self.update()

            
    # def addPoint2NewContour(self, p_image):
    #     p_last = self.np2QPoint(self.NewContour.getLastPoint())
    #     if (p_image != p_last):
    #         self.NewContour.addPoint(self.QPoint2np(p_image))
    #         self.addline(p_last,p_image)
            
    # def prepareNewContour(self):
    #     width = self.image().width()
    #     height = self.image().height()
    #     self.sketch = np.zeros((height, width, 1), np.uint8)
    #     Contour.drawContoursToImage(self.sketch, self.Contours.getShapesOfClass_x(self.parent.activeClass()))
    
    # def getFinalContours(self):
    #     if self.sketch is None:
    #         return
    #     contours = Contour.extractContoursFromImage(self.sketch, not self.parent.allowInnerContours)

    #     # delete changed contours
    #     old_contours = self.Contours.getShapesOfClass_x(self.parent.activeClass())
    #     changedcontours = Contour.getContoursNotinListOfContours(old_contours,contours)
    #     self.Contours.deleteShapes(changedcontours)
    #     for c in contours:
    #         c.setClassLabel(self.parent.activeClass())
            
    #     self.Contours.addShapes(contours)
    #     self.redrawImage()
        
    def checkForChangedContours(self):
        self.painter.checkForChanges()
        # self.prepareNewContour()
        # self.getFinalContours()
        # self.redrawImage()
        
    # def finishNewContour(self, delete = False):
    #     if self.NewContour is not None:
    #         self.NewContour.closeContour()
    #         i = 0 if delete else 1
    #         cv2.drawContours(self.sketch, [self.NewContour.points], 0, (i), -1)  
    #     self.getFinalContours()
    #     self.NewContour = None
        
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
        if self.parent.CurrentFilePath() is None:
            return
        self.rawimage = self.getCurrentPixmap()
        self.clearContours()
        self.getContours()
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
        
    def SaveCurrentContours(self):
        self.painter.save()
        # if not self.Contours.empty():
        #     self.parent.ensureLabelFolder()
        #     self.Contours.save(self.parent.CurrentLabelFullName())
        # else:
        #     self.parent.deleteLabel()
            
    def getContours(self):
        self.painter.load()
        # self.clearContours()
        # if self.parent.CurrentLabelFullName() is not None and os.path.exists(self.parent.CurrentLabelFullName()):
        #     self.Contours.load(self.parent.CurrentLabelFullName())

    
    def redrawImage(self):
        if self.rawimage is not None:
            self.displayedimage.setPixmap(self.rawimage.copy())
            self.drawcontours()
            
    def drawcontours(self):
        if self.image() is None:
            return
        
        self.painter.draw()
        # for c in self.Contours.shapes:  
        #     self.drawcontour(c)
            
        # for p in self.Points.shapes:  
        #     self.drawpoint(p)

        self.updateImage()
        self.parent.numOfContoursChanged()
        
    # def drawpoint(self, p):
    #     painter = self.getPainter()
    #     color = self.parent.ClassColor(p.classlabel)
    #     self.setPainterColor(painter, color)
    #     painter.drawPoint(self.Point2QPoint(p))

                
    # def drawcontour(self, contour):     
    #     painter = self.getPainter()
    #     path = QPainterPath()
    #     if contour.points is not None:
    #         for i in range(contour.numPoints()):
    #             if i == 0:
    #                 path.moveTo(self.np2QPoint(contour.points[i].reshape(2,1)))
    #             else:
    #                 path.lineTo(self.np2QPoint(contour.points[i].reshape(2,1)))
    #         path.lineTo(self.np2QPoint(contour.points[0].reshape(2,1)))
    #         color = self.parent.ClassColor(contour.classlabel)
    #         innerpath = QPainterPath()
    #         for c in contour.innercontours:
    #             for i in range(len(c)):
    #                 if i == 0:
    #                     p = self.np2QPoint(c[i].reshape(2,1))
    #                     innerpath.moveTo(self.np2QPoint(c[i].reshape(2,1)))
    #                 else:
    #                     innerpath.lineTo(self.np2QPoint(c[i].reshape(2,1)))
    #             innerpath.lineTo(p)
    #         path = path.subtracted(innerpath)

    #         self.setPainterColor(painter, color)
    #         self.setColorTransparency(painter, color, transparency = 0)
    #         painter.drawPath(path)
    #         self.drawcontouraccessories(contour, painter)
            
        
    # def drawcontouraccessories(self, contour, painter = None):
    #     color = self.parent.ClassColor(contour.classlabel)
    #     if painter is None:
    #         painter = self.getPainter(color = color)
        
    #     if self.drawContourNumber:
    #         contournumber = self.Contours.getShapeNumber(contour)
    #         painter.drawText(self.getLabelNumPosition(contour) , str(contournumber))
            
    #     if self.drawSkeleton:
    #         skeleton = contour.getSkeleton()         
            
    #         skelpath = QPainterPath()
    #         if skeleton is not None:
    #             for cnt in skeleton:
    #                 i = 0
    #                 for c in cnt:         
    #                     if i == 0:
    #                         skelpath.moveTo(c[0][0], c[0][1])
    #                     else:
    #                         skelpath.lineTo(c[0][0], c[0][1])
    #                     i = i+1
    #             col = color.darker(65)
    #             self.setPainterColor(painter, col)
    #             self.setColorTransparency(painter, color, transparency = -1)
    #             painter.drawPath(skelpath)

        
    # def getLabelNumPosition(self, contour):
    #     bp = contour.getBottomPoint()
    #     pos = QPoint(bp[0],bp[1])+QPoint(0,self.FontSize + 10)
    #     if pos.y() > self.image().height():
    #         pos.setY(self.image().height() - self.FontSize - 5)
    #     if pos.x() > self.image().width():
    #         pos.setX(self.image().width() - self.FontSize - 5)
    #     if pos.x() < 0:
    #         pos.setX(5)
    #     return pos

    # def getPainter(self, color = None):
    #     if color is None:
    #         color = self.parent.ClassColor(self.parent.activeClass())
    #     # painter objects can only exist once per QWidget
    #     p = QPainter(self.image())
    #     color.setAlpha(255)    
    #     p.setPen(QPen(color, self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))

    #     p.setBrush(QBrush(QColor(color),Qt.SolidPattern))
    #     p.setFont(QFont("Fixed",self.FontSize))
    #     return p

    # def setColorTransparency(self, painter, color, transparency):
    #     if transparency == 1:        # opaque
    #         color.setAlpha(255)   
    #     elif transparency == -1:       # invisible
    #         color.setAlpha(0)
    #     elif transparency == 0:       # transparent
    #         color.setAlpha(self.ContourTransparency)
    #     painter.setBrush(QBrush(QColor(color),Qt.SolidPattern))

    # def setPainterColor(self, painter, color):
    #     color.setAlpha(255)
    #     painter.setPen(QPen(color, self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
    #     painter.setBrush(QBrush(QColor(color),Qt.SolidPattern))
        
    def clearContours(self):
        self.painter.clear()
        # self.Contours.clear()
        # self.Points.clear()
        
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
            self.painter.shapes.minSize = self.minContourSize
        elif mode == dlMode.Object_Counting:
            self.painter = ObjectPainter(self)
        # elif mode == dlMode.Classification:
        #     self.painter = ClassPainter(self)
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
    
        