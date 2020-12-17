# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:42:08 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import csv
import os
import glob
import concurrent.futures
from utils.shapes.Contour import loadContours
from utils.shapes.Point import loadPoints
from ui.Tools import canvasTool
from ui.style import styleForm
from ui.ui_utils import DCDButton
from utils.Image import ImageFile
from dl.method.mode import dlMode
import cv2


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Object Detection Settings') 
        width = 250
        height= 120
        Form.setFixedSize(width, height)
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
                
        


class ObjectDetectionResultsWindow(QMainWindow, Window):
    def __init__(self, parent,parentwindow):
        super(ObjectDetectionResultsWindow, self).__init__(parent)
        self.parent = parent
        self.parentwindow = parentwindow
        self.setupUi(self)
        
            
    def saveObjects(self, writer, labels):
        header = ['image name'] + ['number of objects'] + ['object type']
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
        
        for label in labels:
            if isinstance(label, tuple):
                name = os.path.splitext(os.path.basename(label[0]))[0][:-3] + str(int(label[1])+1)
                label = label[0]
            else:
                name = os.path.splitext(os.path.basename(label))[0]

            points,_ = loadPoints(label)
            pointlabels = [x.classlabel for x in points]
            for i in range(1,max(pointlabels)+1):
                row = [name] + [pointlabels.count(i)] + [i]
                writer.writerow(row) 
        
            

            
        
                   
        
        

            

            
