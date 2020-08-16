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


class ContourPainter(Painter):
    def __init__(self, canvas):
        super(ContourPainter,self).__init__(canvas)
        self.tools.append(canvasTool.assign)


    def draw(self):
        super(ContourPainter, self).draw()

        # we can not use a qpainterpath for all contours 
        # as the order of drawing is important 
        # and color has to be the same for the path
        # update: additionally, path.subtracted does not work for multiple objects
        for c in self.shapes.shapes:  
            if c.classlabel != 0:
                self.drawcontour(c)             
        # self.canvas.image().save('export/' + self.canvas.parent.files.CurrentLabelName()+'.tif' , 'TIF')

    def enableDrawBackgroundMode(self):
        super(ContourPainter, self).enableDrawBackgroundMode()


    def disableDrawBackgroundMode(self):
        super(ContourPainter, self).enableDrawBackgroundMode()
        self.tools.append(canvasTool.assign)
    
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
