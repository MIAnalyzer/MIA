# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 13:18:35 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.painter.Painter import Painter

import os
from ui.Tools import canvasTool
from utils.shapes.ImageLabel import ImageLabel, saveImageLabel, loadImageLabel

class ImageLabelPainter(Painter):
    def __init__(self, canvas):
        super(ImageLabelPainter,self).__init__(canvas)
        self.tools.append(canvasTool.drag)
        self.tools.append(canvasTool.setimageclass)
        self.tools.append(canvasTool.setallimagesclass)
        self.tools.append(canvasTool.assignimageclass)
        self.imagelabel = None

    def enableDrawBackgroundMode(self):
        super(ImageLabelPainter, self).enableDrawBackgroundMode()
        self.tools.append(canvasTool.setimageclass)
        self.tools.append(canvasTool.setallimagesclass)
        self.tools.append(canvasTool.assignimageclass)

    def disableDrawBackgroundMode(self):
        super(ImageLabelPainter, self).disableDrawBackgroundMode()
        self.tools.append(canvasTool.setimageclass)
        self.tools.append(canvasTool.setallimagesclass)
        self.tools.append(canvasTool.assignimageclass)

    def draw(self):
        super(ImageLabelPainter, self).draw()
        self.drawLabel()
        
    def clearFOV(self):
        # do nothing
        pass
    
    def addShapes(self,shapes):
        self.addImageLabel(shapes.classlabel)
    
    def clear(self):
        super(ImageLabelPainter, self).clear()
        self.imagelabel = None

    def addImageLabel(self,classlabel=None):
        if classlabel == None:
            classlabel = self.canvas.parent.activeClass()
        self.imagelabel = ImageLabel(classlabel)
        self.canvas.redrawImage()

    def load(self):
        self.clear()
        if self.canvas.parent.files.CurrentLabelPath() is not None and os.path.exists(self.canvas.parent.files.CurrentLabelPath()):    
            self.imagelabel, background = loadImageLabel(self.canvas.parent.files.CurrentLabelPath())
            if background:
                self.contours.addShapes(background)

    def save(self):
        if len(self.trackChanges) > self.maxChanges:
            self.trackChanges.pop(0)
        self.trackChanges.append((self.imagelabel, self.backgroundshapes.copy()))
        self.canvas.parent.checkIfUndoPossible()

        if self.imagelabel or not self.contours.empty():
            self.canvas.parent.ensureLabelFolder()
            saveImageLabel(self.imagelabel, self.canvas.parent.files.CurrentLabelPath(), background = self.backgroundshapes)
        else:
            self.canvas.parent.deleteLabel()

    def undo(self):
        if len(self.trackChanges) < 2:
            return
        self.clear()
        self.imagelabel, background = self.trackChanges[-2]
        self.trackChanges.pop()
        self.trackChanges.pop()
        if background:
            self.contours.addShapes(background)
        self.canvas.parent.checkIfUndoPossible()
        
    def checkForChanges(self):
        # to do
        self.canvas.redrawImage()
        
    def drawLabel(self):
        if not self.imagelabel:
            return
        painter = self.getPainter()
        color = self.canvas.parent.ClassColor(self.imagelabel.classlabel)
        self.setPainterColor(painter, color)
        w = self.canvas.image().width()
        h = self.canvas.image().height()
        p1 = QPoint(0,0)
        p2 = QPoint(w-1,0)
        p3 = QPoint(w-1,h-1)
        p4 = QPoint (0,h-1)
        painter.drawLine(p1, p2)
        painter.drawLine(p2, p3)
        painter.drawLine(p3, p4)
        painter.drawLine(p4, p1)
