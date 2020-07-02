# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:11:55 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_Painter import Painter
import utils.Point as Point
import os
from ui.Tools import canvasTool

class ObjectPainter(Painter):
    def __init__(self, canvas):
        super(ObjectPainter,self).__init__(canvas)
        self.shapes = Point.Points()
        self.tools.append(canvasTool.point)
        self.tools.append(canvasTool.shift)
        self.tools.append(canvasTool.assign)
        self.tools.append(canvasTool.delete)
        
    def draw(self):
        for p in self.shapes.shapes:  
            self.drawpoint(p)
    
    def clear(self):
        self.shapes.clear()

    def load(self):
        self.clear()
        if self.canvas.parent.CurrentLabelFullName() is not None and os.path.exists(self.canvas.parent.CurrentLabelFullName()):
            self.shapes.load(self.canvas.parent.CurrentLabelFullName())
        
    def save(self):
        if not self.shapes.empty():
            self.canvas.parent.ensureLabelFolder()
            self.shapes.save(self.canvas.parent.CurrentLabelFullName())
        else:
            self.canvas.parent.deleteLabel()
        
    def checkForChanges(self):
        # to do
        self.canvas.redrawImage()
        
        
    def addPoint(self, point, classlabel=None):
        if classlabel == None:
            classlabel = self.canvas.parent.activeClass()
        self.shapes.addShape(Point.Point(classlabel,self.canvas.QPoint2npPoint(point)))
        self.canvas.redrawImage()

    def drawpoint(self,p):
        painter = self.getPainter()
        color = self.canvas.parent.ClassColor(p.classlabel)
        self.setPainterColor(painter, color)
        
        qp = self.canvas.Point2QPoint(p)
        p1 = QPoint(qp.x() - self.canvas.FontSize, qp.y())
        p2 = QPoint(qp.x() + self.canvas.FontSize, qp.y())
        p3 = QPoint(qp.x(), qp.y() - self.canvas.FontSize)
        p4 = QPoint(qp.x(), qp.y() + self.canvas.FontSize)
        
        painter.drawLine(p1, p2)
        painter.drawLine(p3, p4)

        if self.canvas.drawShapeNumber:
            pointnumber = self.shapes.getShapeNumber(p)
            painter.drawText(self.getLabelNumPosition(p) , str(pointnumber))
            
    def getLabelNumPosition(self, point):
        coo = point.coordinates
        pos = QPoint(coo[0],coo[1])+QPoint(10,self.canvas.FontSize+10)
        if pos.y() > self.canvas.image().height():
            pos.setY(self.canvas.image().height() - self.canvas.FontSize - 5)
        if pos.x() > self.canvas.image().width():
            pos.setX(self.canvas.image().width() - self.canvas.FontSize - 5)
        if pos.x() < 0:
            pos.setX(5)
        return pos