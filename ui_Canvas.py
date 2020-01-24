# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:44:53 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os

import Tools
from Contour import *
from Tools import canvasTool
import numpy as np

class Canvas(QGraphicsView):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.colors = []
        self.pen_size = 3
        self.bgcolor = QColor(0,0,0)
        self.image = None
        self.bufferimage = None
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        self.Contours = Contours()
        self.activeContour = None
        self.enableToggleTools = True
        self.FontSize = 18
        
        self.setCursor(Qt.OpenHandCursor)
        self.displayedimage = QGraphicsPixmapItem()
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.displayedimage)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

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
      
    def addline(self, p1 ,p2):
        p = self.getPainter()
        p.drawLine(p2, p1)
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def addcircle(self, center , radius):
        p = self.getPainter()
        p.setBrush(QBrush(self.parent.ClassColor(self.parent.activeClass()), Qt.SolidPattern))
        p.drawEllipse(center, radius, radius)
        p.setBrush(QBrush(self.parent.ClassColor(self.parent.activeClass()), Qt.NoBrush))
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def addPoint2Contour(self, p_image):
        if (p_image != self.activeContour.getLastPoint()):
            self.activeContour.addPoint(p_image)
            return True
        else:
            return False
            
    def finishContour(self):
        if self.activeContour is None:
            return
        self.activeContour.closeContour()
        if (self.activeContour.isValid()):
            self.Contours.addContour(self.activeContour)
            paint = self.getPainter()
            paint.drawText(self.getLabelNumPosition(self.activeContour) , str(self.Contours.numOfContours()))
            self.displayedimage.setPixmap(self.image)
            self.parent.numOfContoursChanged()
            self.update()

        else:
            self.redrawImage()
        self.activeContour = None
        
    def fitInView(self, scale=True):
        rect = QRectF(self.displayedimage.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                self.zoomstep = 0

    def reset(self, width, height):
        self.displayedimage.setPixmap(QPixmap(width, height))
        self.displayedimage.pixmap().fill(self.bgcolor)
        
    def newImage(self):
        self.bufferimage = QPixmap(self.parent.files[self.parent.currentImage])
        self.clearContours()
        self.getContours()
        self.redrawImage()
        self.zoomstep = 0
        self.fitInView()
        
    def hasImage(self):
        return self.image is not None
        
    def zoom(self):
        width = self.geometry().width()
        height = self.geometry().height()
        factor = 1 + self.zoomstep/10 
        pix = self.image.scaled(width * factor, height * factor, Qt.KeepAspectRatio)
        self.displayedimage.setPixmap(pix)
        
    def createLabelFromContours(self):
        if not self.Contours.empty():
            self.Contours.saveContours(self.parent.CurrentLabelFullName())
              
    def getContours(self):
        self.clearContours()
        if os.path.exists(self.parent.CurrentLabelFullName()):
            self.Contours.loadContours(self.parent.CurrentLabelFullName())
            self.drawcontours()
    
    def redrawImage(self):
        if self.bufferimage is not None:
            self.image = self.bufferimage.copy()
            self.displayedimage.setPixmap(self.image)
            self.drawcontours()
    
    def drawcontours(self):
        if self.image is None:
            return
        paint = self.getPainter()
        contournum = 0
        for c in self.Contours.contours:  
            contournum += 1
            path = QPainterPath()
            if c.points is not None:
                for i in range(c.numPoints()):
                    if i == 0:
                        path.moveTo(np2QPoint(c.points[i].reshape(2,1)))
                    else:
                        path.lineTo(np2QPoint(c.points[i].reshape(2,1)))
                path.lineTo(np2QPoint(c.points[0].reshape(2,1)))
                self.setPainterColor(paint, self.parent.ClassColor(c.classlabel))
                paint.drawPath(path)
                paint.drawText(self.getLabelNumPosition(c) , str(contournum))

             
        self.displayedimage.setPixmap(self.image)
        self.parent.numOfContoursChanged()
        self.update()
        
    def getLabelNumPosition(self, contour):
        pos = contour.getBottomPoint()+QPoint(0,self.FontSize + 10)
        if pos.y() > self.image.height():
            pos.setY(self.image.height() - self.FontSize - 5)
        if pos.x() > self.image.width():
            pos.setX(self.image.width() - self.FontSize - 5)
        if pos.x() < 0:
            pos.setX(5)
        return pos

    def getPainter(self):
        # painter objects can only exist once per QWidget
        p = QPainter(self.image)
        p.setPen(QPen(self.parent.ClassColor(self.parent.activeClass()), self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        p.setFont(QFont("Fixed",self.FontSize))
        return p
        
    def setPainterColor(self, painter, color):
        painter.setPen(QPen(color, self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        
    def clearContours(self):
        self.Contours.clear()
        
    def f2intPoint(self, p):
        # we do not round here!
        # by using floor it is ensured that the mouse arrow head is inside the selected pixel
        p.setX(int(p.x()))
        p.setY(int(p.y()))
        return p
    
    def setnewTool(self, tool):
        self.lasttool = self.tool  
        self.parent.ExtendSettings.hide()
        if tool == canvasTool.drag.name:
            self.tool = Tools.DragTool(self)
        if tool == canvasTool.draw.name:
            self.tool = Tools.DrawTool(self)
        if tool == canvasTool.assign.name:
            self.tool = Tools.AssignTool(self)
        if tool == canvasTool.expand.name:
            self.tool = Tools.ExtendTool(self)
            self.parent.ExtendSettings.show()
        if tool == canvasTool.delete.name:
            self.tool = Tools.DeleteTool(self)
        if tool == canvasTool.poly.name:
            self.tool = Tools.PolygonTool(self)
        
        self.setCursor(self.tool.Cursor())
        

    def toggleDrag(self):
        if not self.enableToggleTools:
            return
            
        if self.tool.type == canvasTool.drag:
            self.setnewTool(self.lasttool.type.name)
        else:
            self.setnewTool(canvasTool.drag.name)