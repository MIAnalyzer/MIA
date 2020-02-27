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
from utils.Contour import *


class Canvas(QGraphicsView):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.colors = []
        self.pen_size = 3
        self.bgcolor = QColor(0,0,0)
        self.image = None
        self.rawimage = None
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        self.Contours = Contours()
        self.activeContour = None
        self.enableToggleTools = True
        self.FontSize = 18
        self.drawContourNumber = True
        self.drawSkeleton = False
        
        
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
      
    def addline(self, p1 ,p2, color = None):
        p = self.getPainter(color)
        p.drawLine(p2, p1)
        self.updateImage()
        
    def addTriangle(self, point, size, color = None):
        psize = self.pen_size
        self.pen_size = 1
        p = self.getPainter(color, True)
        polygon = QPolygonF()
        polygon.append(point)
        point2 = QPointF(point.x() + size/2, point.y() - size)
        polygon.append(point2)
        point3 = QPointF(point.x() - size/2, point.y() - size)
        polygon.append(point3)
        p.drawPolygon(polygon)
        self.pen_size = psize
        self.updateImage()
        
    def addcircle(self, center , radius, color = None):
        p = self.getPainter(color, True)
        p.drawEllipse(center, radius, radius)
        self.updateImage()
        
    def updateImage(self):
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
        if (self.activeContour.isValid(self.Contours.minSize)):
            self.Contours.addContour(self.activeContour)
            self.drawcontouraccessories(self.activeContour)
            self.updateImage()
            self.parent.numOfContoursChanged()
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
        
    def ReloadImage(self):
        if self.parent.CurrentFilePath() is None:
            return
        self.rawimage = QPixmap(self.parent.CurrentFilePath())
        self.clearContours()
        self.getContours()
        self.redrawImage()
        self.zoomstep = 0
        self.fitInView()
        
    def hasImage(self):
        return self.image is not None
    
    def setScale(self,scale_ppmm):
        self.scale_pixel_per_mm = scale_ppmm
        if int(self.scale_pixel_per_mm+0.5) != int(self.parent.results_form.LEScale.text()):
            self.parent.results_form.LEScale.setText(str(int(self.scale_pixel_per_mm+0.5)))
        
    def zoom(self):
        width = self.geometry().width()
        height = self.geometry().height()
        factor = 1 + self.zoomstep/10 
        pix = self.image.scaled(width * factor, height * factor, Qt.KeepAspectRatio)
        self.displayedimage.setPixmap(pix)
        
    def createLabelFromContours(self):
        if not self.Contours.empty():
            self.Contours.saveContours(self.parent.CurrentLabelFullName())
        else:
            self.parent.deleteLabel()
            
    def getContours(self):
        self.clearContours()
        if self.parent.CurrentLabelFullName() is not None and os.path.exists(self.parent.CurrentLabelFullName()):
            self.Contours.loadContours(self.parent.CurrentLabelFullName())
    
    def redrawImage(self):
        if self.rawimage is not None:
            self.image = self.rawimage.copy()
            self.displayedimage.setPixmap(self.image)
            self.drawcontours()
    
    def drawcontours(self):
        if self.image is None:
            return

        for c in self.Contours.contours:  
            self.drawcontour(c)

        self.updateImage()
        self.parent.numOfContoursChanged()
        
    def drawcontour(self, contour):
        painter = self.getPainter()
        path = QPainterPath()
        if contour.points is not None:
            for i in range(contour.numPoints()):
                if i == 0:
                    path.moveTo(np2QPoint(contour.points[i].reshape(2,1)))
                else:
                    path.lineTo(np2QPoint(contour.points[i].reshape(2,1)))
            path.lineTo(np2QPoint(contour.points[0].reshape(2,1)))
            color = self.parent.ClassColor(contour.classlabel)
            self.setPainterColor(painter, color)
            painter.drawPath(path)
            self.drawcontouraccessories(contour, painter)
            
        
    def drawcontouraccessories(self, contour, painter = None):
        color = self.parent.ClassColor(contour.classlabel)
        if painter is None:
            painter = self.getPainter(color = color)
        
        if self.drawContourNumber:
            contournumber = self.Contours.getContourNumber(contour)
            painter.drawText(self.getLabelNumPosition(contour) , str(contournumber))
            
        if self.drawSkeleton:
            skeleton = contour.getSkeleton()         
            i = 0
            skelpath = QPainterPath()
            if skeleton is not None:
                for c in skeleton:
                    if i == 0:
                        skelpath.moveTo(c[0][0], c[0][1])
                    else:
                        skelpath.lineTo(c[0][0], c[0][1])
                    i = i+1
                col = color.darker(65)
                self.setPainterColor(painter, col)
                painter.drawPath(skelpath)

        
    def getLabelNumPosition(self, contour):
        pos = contour.getBottomPoint()+QPoint(0,self.FontSize + 10)
        if pos.y() > self.image.height():
            pos.setY(self.image.height() - self.FontSize - 5)
        if pos.x() > self.image.width():
            pos.setX(self.image.width() - self.FontSize - 5)
        if pos.x() < 0:
            pos.setX(5)
        return pos

    def getPainter(self, color = None, filled = False,):
        if color is None:
            color = self.parent.ClassColor(self.parent.activeClass())
        # painter objects can only exist once per QWidget
        p = QPainter(self.image)
        if filled == True:
            p.setBrush(QBrush(QColor(color),Qt.SolidPattern))
        else:
            p.setBrush(QBrush(Qt.NoBrush))
        p.setPen(QPen(color, self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
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
        if tool == canvasTool.scale.name:
            self.tool = Tools.ScaleTool(self)
        
        self.parent.StatusMode.setText('Current Mode: ' + self.tool.Text) 
        self.setCursor(self.tool.Cursor())
        

    def toggleDrag(self):
        if not self.enableToggleTools:
            return
            
        if self.tool.type == canvasTool.drag:
            self.setnewTool(self.lasttool.type.name)
        else:
            self.setnewTool(canvasTool.drag.name)