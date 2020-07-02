# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:37:52 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


from abc import ABC, abstractmethod

class Painter(ABC):
    def __init__(self, canvas):
        self.canvas = canvas
        self.shapes = None
        self.tools = []
    
    @abstractmethod
    def draw(self):
        pass
    
    @abstractmethod
    def clear(self):
        pass
        
    @abstractmethod
    def load(self):
        pass
        
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def checkForChanges(self):
        pass
        
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
