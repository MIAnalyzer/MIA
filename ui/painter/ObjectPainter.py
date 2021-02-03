# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:11:55 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.painter.Painter import Painter
import utils.shapes.Point as Point
import os
from ui.Tools import canvasTool
import copy

class ObjectPainter(Painter):
    def __init__(self, canvas):
        super(ObjectPainter,self).__init__(canvas)
        self.points = Point.Points()
        self.backgroundmode = False
        self.tools.append(canvasTool.objectnumber)
        self.tools.append(canvasTool.objectcolor)
        self.tools.append(canvasTool.deleteobject)


    @property
    def shapes(self):
        if self.backgroundmode:
            return self.contours
        else:
            return self.points
        
    def clearFOV(self):
        to_del = []
        fov = self.canvas.getFieldOfViewRect()
        for p in self.points.shapes:
            if fov.x() < p.coordinates[0] < fov.x()+fov.width() and fov.y() < p.coordinates[1] < fov.y()+fov.height():
                to_del.append(p)
        self.points.deleteShapes(to_del)
        self.canvas.redrawImage()
        
    def addShapes(self,shapes):
        self.points.addShapes(shapes)

    def enableDrawBackgroundMode(self):
        super(ObjectPainter, self).enableDrawBackgroundMode()
        self.backgroundmode = True

    def disableDrawBackgroundMode(self):
        super(ObjectPainter, self).disableDrawBackgroundMode()
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.point)
        self.tools.append(canvasTool.shift)
        self.tools.append(canvasTool.assign)
        self.tools.append(canvasTool.assist)
        self.tools.append(canvasTool.delete)
        self.tools.append(canvasTool.objectnumber)
        self.tools.append(canvasTool.objectcolor)
        self.tools.append(canvasTool.deleteobject)
        self.backgroundmode = False
        
    def draw(self):
        super(ObjectPainter, self).draw()
        # for x in range(self.canvas.parent.NumOfClasses()):
        for x in range(self.points.getMaxClass()):    
            self.drawClass(x+1)

    def drawClass(self, x):  
        painter = self.getPainter()
        path = QPainterPath()
        shapes = self.points.getShapesOfClass_x(x)
        color = self.canvas.parent.ClassColor(x)
        self.setPainterColor(painter, color)
        drawall = True

        for p in shapes:
            qp = self.canvas.Point2QPoint(p)
            p1 = QPoint(qp.x() - self.canvas.FontSize, qp.y())
            p2 = QPoint(qp.x() + self.canvas.FontSize, qp.y())
            p3 = QPoint(qp.x(), qp.y() - self.canvas.FontSize)
            p4 = QPoint(qp.x(), qp.y() + self.canvas.FontSize)
                     
            path.moveTo(p1)
            path.lineTo(p2)
            path.moveTo(p3)
            path.lineTo(p4)

            if self.canvas.parent.colorCodeObjects():
                if p.objectNumber != -1:
                    color = self.getObjectColor(p.objectNumber)
                else:
                    color = self.canvas.parent.ClassColor(x)
                self.setPainterColor(painter, color)
                painter.drawPath(path) 
                path = QPainterPath()
                drawall = False

            self.drawtrack(p, painter, color)
            if self.canvas.drawShapeNumber:
                if not drawall:
                    color = self.canvas.parent.ClassColor(x)
                    self.setPainterColor(painter, color)
                pointnumber = self.points.getShapeNumber(p)
                painter.drawText(self.getLabelNumPosition(p) , str(pointnumber))
            
        if drawall:
            painter.drawPath(path)   
    
    def clear(self):
        super(ObjectPainter, self).clear()
        self.points.clear()

    def load(self):
        self.clear()
        path = self.canvas.parent.files.CurrentLabelPath()
        if path is not None and os.path.exists(path):
            points, background = Point.loadPoints(path)
            self.points.addShapes(points)
            if background:
                self.contours.addShapes(background)
        
    def save(self):
        if len(self.trackChanges) > self.maxChanges:
            self.trackChanges.pop(0)
        self.trackChanges.append((copy.deepcopy(self.points.shapes), copy.deepcopy(self.backgroundshapes)))
        self.canvas.parent.checkIfUndoPossible()
        if not self.points.empty() or not self.contours.empty():
            self.canvas.parent.ensureLabelFolder()
            Point.savePoints(self.points.shapes, self.canvas.parent.files.CurrentLabelPath(), background = self.backgroundshapes)
        else:
            self.canvas.parent.deleteLabel()

    def undo(self):
        if len(self.trackChanges) < 2:
            return
        self.clear()
        points, background = self.trackChanges[-2]
        self.trackChanges.pop()
        self.trackChanges.pop()
        self.points.addShapes(points)
        if background:
            self.contours.addShapes(background)
        self.canvas.parent.checkIfUndoPossible()

        
    def checkForChanges(self):
        # to do
        self.canvas.redrawImage()
        
        
    def addPoint(self, point, classlabel=None):
        if classlabel == None:
            classlabel = self.canvas.parent.activeClass()
        
        self.points.addShape(Point.Point(classlabel,self.canvas.QPoint2npPoint(point)))  
        self.canvas.redrawImage()

    def autodetection(self):
        if self.canvas.hasImage():
            self.clearFOV()
            image = self.canvas.getFieldOfViewImage()
            fov = self.canvas.getFieldOfViewRect()
            # not using self.dl.dataloader.readImage() here as a 8-bit rgb image is required and no extra preprocessing
    
            if image is None:
                self.canvas.parent.PopupWarning('No image loaded')
                return
            try:
                prediction = self.canvas.parent.dl.AutoSegment(image)
            except:
                self.canvas.parent.PopupWarning('Auto prediction not possible')
                return
            shapes = Point.extractPointsFromContours(prediction, 5, offset = (int(fov.x()),int(fov.y())))
            for s in shapes:
                s.setClassLabel(self.canvas.parent.activeClass())
            self.points.addShapes(shapes)
            self.checkForChanges()

            
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