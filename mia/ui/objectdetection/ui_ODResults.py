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

        styleForm(Form)
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        self.CBExPosition = QCheckBox("Export Positions",self.centralWidget)
        self.CBExPosition.setObjectName('ExportPositions')
        self.CBExPosition.setChecked(True)
        self.CBExPosition.setToolTip('Check to export positions of detected objects')
        
        self.vlayout.addWidget(self.CBExPosition)
        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())
                

class ObjectDetectionResultsWindow(QMainWindow, Window):
    def __init__(self, parent,parentwindow):
        super(ObjectDetectionResultsWindow, self).__init__(parent)
        self.parent = parent
        self.parentwindow = parentwindow
        self.setupUi(self)
        
            
    def saveObjects(self, writer, images, labels):

        if self.parent.TrackingModeEnabled:
            self.saveTrackResults(writer, images, labels)
        else:
            self.saveDetectionResults(writer, images, labels)

    def saveDetectionResults(self, writer, images, labels):
        header = ['image name'] + ['frame number']
        if self.CBExPosition.isChecked():
            header = header + ['object number'] + ['object type'] + ['position'] 
        else:
            header = header + ['number of objects'] + ['object type']
            
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
        
        for image, label in zip(images,labels):
            name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(image),withfileextension=True)
            points,_ = loadPoints(self.parent.files.convertIfStackPath(label))
            if self.parent.files.isStackLabel(label):
                frame = self.parent.files.getFrameNumber(label) + 1
            else:
                frame = 1
                

            if self.CBExPosition.isChecked():
                for i,p in enumerate(points):
                    num = p.objectNumber if p.objectNumber >= 0 else i+1
                    row = [name] + [frame] + [num] + [p.classlabel] + [p.getPosition()]
                    writer.writerow(row) 
            else:
                pointlabels = [x.classlabel for x in points]
                for i in range(1,max(pointlabels)+1):
                    row = [name] + [frame] + [pointlabels.count(i)] + [i]
                    writer.writerow(row) 


    def saveTrackResults(self, writer, images, labels):
        header = ['object number'] + ['time point'] + ['image name'] + ['frame number'] + ['object type'] + ['position']
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)

        f1 = lambda x: x.classlabel
        f2 = lambda x: x.getPosition()

        pnt_data = (f1,f2)

        points = [{} for x in range(self.parent.tracking.timepoints)]
        for i,l in zip(images,labels):
            name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(i),withfileextension=True)
            point_s,_ = loadPoints(self.parent.files.convertIfStackPath(l))

            if self.parent.tracking.stackMode and self.parent.files.isStackLabel(l):
                tp = self.parent.files.getFrameNumber(l)
            else:
                tp = self.parent.tracking.getTimePointFromImageName(self.parent.files.convertIfStackPath(i))
            if self.parent.files.isStackLabel(l):
                frame = self.parent.files.getFrameNumber(l) + 1
            else:
                frame = 1
            points[tp] = {x.objectNumber: [name] + [frame] + [f(x) for f in pnt_data] for x in point_s}
            self.parent.emitProgress()

        for n in self.parent.tracking.getNumbersOfTrackedObjects():
            for t in self.parent.tracking.getObjectOccurences(n):
                try:
                    row = [n] + [t+1] + points[t][n]
                    writer.writerow(row) 
                except:
                    pass