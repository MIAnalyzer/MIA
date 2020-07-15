# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:55:22 2020

@author: Koerber
"""



from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import os
import numpy as np
import cv2
from ui.ui_Painter import Painter
import utils.Contour as Contour
from ui.Tools import canvasTool

class ContourPainter(Painter):
    def __init__(self, canvas):
        super(ContourPainter,self).__init__(canvas)
        self.shapes = Contour.Contours()
        self.NewContour = None
        self.sketch = None
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.draw)
        self.tools.append(canvasTool.assign)
        self.tools.append(canvasTool.extend)
        self.tools.append(canvasTool.poly)
        self.tools.append(canvasTool.delete)

    def draw(self):
        bg = self.shapes.getShapesOfClass_x(0)
        for c in bg:  
            self.drawcontour(c)

        for c in self.shapes.shapes:  
            if c.classlabel != 0:
                self.drawcontour(c) 
        # self.canvas.image().save('export/' + self.canvas.parent.files.CurrentLabelName()+'.tif' , 'TIF')
    
    def clear(self):
        self.shapes.clear()

    def load(self):
        self.clear()
        if self.canvas.parent.files.CurrentLabelPath() is not None and os.path.exists(self.canvas.parent.files.CurrentLabelPath()):
            self.shapes.load(self.canvas.parent.files.CurrentLabelPath())
        
    def save(self):
        if not self.shapes.empty():
            self.canvas.parent.ensureLabelFolder()
            self.shapes.save(self.canvas.parent.files.CurrentLabelPath())
        else:
            self.canvas.parent.deleteLabel()
        
    def checkForChanges(self):
        self.prepareNewContour()
        self.getFinalContours()

        
    def addPoint2NewContour(self, p_image):
        p_last = self.canvas.np2QPoint(self.NewContour.getLastPoint())
        if (p_image != p_last):
            self.NewContour.addPoint(self.canvas.QPoint2np(p_image))
            self.addline(p_last,p_image)
            
    def prepareNewContour(self):
        width = self.canvas.image().width()
        height = self.canvas.image().height()
        self.sketch = np.zeros((height, width, 1), np.uint8)
        Contour.drawContoursToImage(self.sketch, self.shapes.getShapesOfClass_x(self.canvas.parent.activeClass()))
    
    def getFinalContours(self):
        if self.sketch is None:
            return
        contours = Contour.extractContoursFromImage(self.sketch, not self.canvas.parent.allowInnerContours)

        # delete changed contours
        old_contours = self.shapes.getShapesOfClass_x(self.canvas.parent.activeClass())
        changedcontours = Contour.getContoursNotinListOfContours(old_contours,contours)
        self.shapes.deleteShapes(changedcontours)
        for c in contours:
            c.setClassLabel(self.canvas.parent.activeClass())
            
        self.shapes.addShapes(contours)
        self.canvas.redrawImage()
        
    def finishNewContour(self, delete = False):
        if self.NewContour is not None:
            self.NewContour.closeContour()
            i = 0 if delete else 1
            cv2.drawContours(self.sketch, [self.NewContour.points], 0, (i), -1)  
        self.getFinalContours()
        self.NewContour = None
        
        
    def drawcontour(self, contour):     
        painter = self.getPainter()
        path = QPainterPath()
        if contour.points is not None:
            for i in range(contour.numPoints()):
                if i == 0:
                    path.moveTo(self.canvas.np2QPoint(contour.points[i].reshape(2,1)))
                else:
                    path.lineTo(self.canvas.np2QPoint(contour.points[i].reshape(2,1)))
            path.lineTo(self.canvas.np2QPoint(contour.points[0].reshape(2,1)))
            color = self.canvas.parent.ClassColor(contour.classlabel)
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

            self.setPainterColor(painter, color)
            self.setColorTransparency(painter, color, transparency = 0)
            painter.drawPath(path)
            self.drawcontouraccessories(contour, painter)
            
        
    def drawcontouraccessories(self, contour, painter = None):
        color = self.canvas.parent.ClassColor(contour.classlabel)
        if painter is None:
            painter = self.getPainter(color = color)
        
        if self.canvas.drawShapeNumber:
            contournumber = self.shapes.getShapeNumber(contour)
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