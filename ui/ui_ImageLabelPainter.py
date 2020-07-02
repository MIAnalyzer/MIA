# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 13:18:35 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_Painter import Painter

import os
from ui.Tools import canvasTool
from utils.ImageLabel import ImageLabel, saveImageLabel, loadImageLabel

class ImageLabelPainter(Painter):
    def __init__(self, canvas):
        super(ImageLabelPainter,self).__init__(canvas)
        self.tools.append(canvasTool.drag)
        self.imagelabel = None

    def draw(self):
        self.drawLabel()
    
    def clear(self):
        self.imagelabel = None

    def addImageLabel(self,classlabel=None):
        if classlabel == None:
            classlabel = self.canvas.parent.activeClass()
        self.imagelabel = ImageLabel(classlabel)
        self.canvas.redrawImage()

    def load(self):
        self.clear()
        if self.canvas.parent.CurrentLabelFullName() is not None and os.path.exists(self.canvas.parent.CurrentLabelFullName()):
            self.imagelabel = loadImageLabel(self.canvas.parent.CurrentLabelFullName())
        
    def save(self):
        if self.imagelabel:
            self.canvas.parent.ensureLabelFolder()
            saveImageLabel(self.imagelabel, self.canvas.parent.CurrentLabelFullName())
        else:
            self.canvas.parent.deleteLabel()
        
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
