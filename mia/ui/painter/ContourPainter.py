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
from ui.painter.Painter import Painter
import utils.shapes.Contour as Contour
from ui.Tools import canvasTool
import copy 

class ContourPainter(Painter):
    def __init__(self, canvas):
        super(ContourPainter,self).__init__(canvas)
        self.tools.append(canvasTool.assign)
        self.tools.append(canvasTool.assist)
        self.tools.append(canvasTool.objectnumber)
        self.tools.append(canvasTool.objectcolor)
        self.tools.append(canvasTool.deleteobject)
        
        # self.mask = None
        self.smartmode = False

        
    @property
    def sketch(self):
        if self.smartmode:
            return self.mask
        else:
            return self.contoursketch
        
    def addShapes(self,shapes):
        self.contours.addShapes(shapes)

    def draw(self):
        super(ContourPainter, self).draw()

        # we can not use a qpainterpath for all contours 
        # as the order of drawing is important 
        # and color has to be the same for the path
        # update: additionally, path.subtracted does not work for multiple objects
        shapes = sorted(self.shapes.shapes, key=lambda x: x.classlabel)
        for c in shapes: 
            if c.classlabel != 0:
                self.drawcontour(c)             

    def enableDrawBackgroundMode(self):
        super(ContourPainter, self).enableDrawBackgroundMode()

    def disableDrawBackgroundMode(self):
        super(ContourPainter, self).enableDrawBackgroundMode()
        self.tools.append(canvasTool.assign)
        self.tools.append(canvasTool.assist)
        self.tools.append(canvasTool.objectnumber)
        self.tools.append(canvasTool.objectcolor)
        self.tools.append(canvasTool.deleteobject)

    def clear(self):
        super(ContourPainter, self).clear()
        self.shapes.clear()

    def load(self):
        self.clear()
        if self.canvas.parent.files.CurrentLabelPath() is not None and os.path.exists(self.canvas.parent.files.CurrentLabelPath()):
            self.shapes.load(self.canvas.parent.files.CurrentLabelPath())
        
    def save(self):
        if len(self.trackChanges) > self.maxChanges:
            self.trackChanges.pop(0)
        self.trackChanges.append(copy.deepcopy(self.shapes.shapes))
        self.canvas.parent.checkIfUndoPossible()
        if not self.shapes.empty():
            self.canvas.parent.ensureLabelFolder()
            self.shapes.save(self.canvas.parent.files.CurrentLabelPath())
        else:
            self.canvas.parent.deleteLabel()

    def undo(self):
        if len(self.trackChanges) < 2:
            return
        self.clear()
        contours = self.trackChanges[-2]
        self.trackChanges.pop()
        self.trackChanges.pop()
        self.shapes.addShapes(contours)
        self.canvas.parent.checkIfUndoPossible()
            
    def clearFOV(self, currentclass = False):
        width = self.canvas.image().width()
        height = self.canvas.image().height()
        self.contoursketch = np.zeros((height, width), np.uint8) 
        Contour.drawContoursToLabel(self.contoursketch, self.contours.shapes, drawbackground=False)
        fov = self.canvas.getFieldOfViewRect()
        if currentclass:
            rect = self.contoursketch[int(fov.y()):int(fov.y()+fov.height()),int(fov.x()):int(fov.x()+fov.width())]
            rect[rect == self.canvas.parent.activeClass()] = 0
            self.contoursketch[int(fov.y()):int(fov.y()+fov.height()),int(fov.x()):int(fov.x()+fov.width())] = rect
        else:
            self.contoursketch[int(fov.y()):int(fov.y()+fov.height()),int(fov.x()):int(fov.x()+fov.width())] = 0    
        contours = Contour.extractContoursFromLabel(self.contoursketch, not self.canvas.parent.allowInnerContours)
        self.contours.clear()
        self.contours.addShapes(contours)
        self.canvas.redrawImage()
        
    def prepareNewContour(self):
        if self.canvas.hasImage():
            width = self.canvas.image().width()
            height = self.canvas.image().height()

            if self.smartmode:
                if self.mask is None:
                    self.mask = np.zeros((height, width), np.uint8) + 2

            super(ContourPainter, self).prepareNewContour()


    def getFinalContours(self):
        if self.sketch is None:
            return
        
        if self.smartmode:
            self.clearFOV(currentclass = True)
            image = self.canvas.getFieldOfViewImage()
            fov = self.canvas.getFieldOfViewRect()
            mask = self.canvas.getFieldOfViewImage(self.mask)
            prediction = self.canvas.parent.dl.SemiAutoSegment(image, mask)
            self.contours.deleteShapes(self.contours.getShapesOfClass_x(self.canvas.parent.activeClass()))
            smartcontours = Contour.extractContoursFromImage(prediction, not self.canvas.parent.allowInnerContours, offset = (int(fov.x()),int(fov.y())))
            Contour.drawContoursToImage(self.contoursketch, smartcontours)  

        super(ContourPainter, self).getFinalContours()

    def assistedSegmentation(self, extremepoints):
        with self.canvas.parent.wait_cursor():
            # do not use fov here because the target image is cropped to network size
            prediction = self.canvas.parent.dl.AssistedSegmentation(self.canvas.parent.getCurrentImage(), np.asarray(extremepoints, dtype = np.int))
            if prediction is None:
                self.canvas.redrawImage()
                return
            shapes = Contour.extractContoursFromLabel(prediction, not self.canvas.parent.allowInnerContours)
            for s in shapes:
                s.setClassLabel(self.canvas.parent.activeClass())
            self.contours.addShapes(shapes)
            self.checkForChanges()
        
    def autosegment(self):
        if self.canvas.hasImage():
            self.clearFOV(currentclass = True)
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
            shapes = Contour.extractContoursFromLabel(prediction, not self.canvas.parent.allowInnerContours, offset = (int(fov.x()),int(fov.y())))
            for s in shapes:
                s.setClassLabel(self.canvas.parent.activeClass())
            self.contours.addShapes(shapes)
            self.checkForChanges()
        
    def checkForChanges(self):
        if self.canvas.hasImage():
            width = self.canvas.image().width()
            height = self.canvas.image().height()
            self.contoursketch = np.zeros((height, width), np.uint8) 
            Contour.drawContoursToLabel(self.contoursketch, self.contours.shapes)
            
            contours = Contour.extractContoursFromLabel(self.contoursketch, not self.canvas.parent.allowInnerContours)
            # delete changed contours
            old_contours = self.contours.shapes
            changedcontours = Contour.getContoursNotinListOfContours(old_contours,contours)
            self.contours.deleteShapes(changedcontours)
                
            self.contours.addShapes(contours)
            self.canvas.redrawImage()
        