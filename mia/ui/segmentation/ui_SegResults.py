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
from utils.shapes.Contour import loadContours, getContourMinIntensity,getContourMeanIntensity, getContourMaxIntensity
from utils.shapes.Point import loadPoints
from ui.Tools import canvasTool
from ui.style import styleForm
from ui.ui_utils import DCDButton
from utils.Image import ImageFile
from dl.method.mode import dlMode
import cv2


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Segmentation Settings') 

        styleForm(Form)
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        
        self.CBSkeleton = QCheckBox("Export Skeleton",self.centralWidget)
        self.CBSkeleton.setObjectName('ExportSkeleton')
        self.CBSkeleton.setChecked(True)
        self.CBSkeleton.setToolTip('Check to export skeleton length')

        self.CBPerimeter = QCheckBox("Export Perimeter",self.centralWidget)
        self.CBPerimeter.setObjectName('Exportperimeter')
        self.CBPerimeter.setChecked(True)
        self.CBPerimeter.setToolTip('Check to export Perimeter length')

        self.CBMin = QCheckBox("Export Min Intensity",self.centralWidget)
        self.CBMin.setObjectName('ExportMin')
        self.CBMin.setChecked(True)
        self.CBMin.setToolTip('Check to export minimum pixel intensity')

        self.CBMean = QCheckBox("Export Mean Intensity",self.centralWidget)
        self.CBMean.setObjectName('ExportMean')
        self.CBMean.setChecked(True)
        self.CBMean.setToolTip('Check to export mean pixel intensity')

        self.CBMax = QCheckBox("Export Max Intensity",self.centralWidget)
        self.CBMax.setObjectName('ExportMax')
        self.CBMax.setChecked(True)
        self.CBMax.setToolTip('Check to export maximum pixel intensity')

        self.vlayout.addWidget(self.CBSkeleton)
        self.vlayout.addWidget(self.CBPerimeter)
        self.vlayout.addWidget(self.CBMin)
        self.vlayout.addWidget(self.CBMean)
        self.vlayout.addWidget(self.CBMax)

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())
                
        


class SegmentationResultsWindow(QMainWindow, Window):
    def __init__(self, parent, parentwindow):
        super(SegmentationResultsWindow, self).__init__(parent)
        self.parent = parent
        self.parentwindow = parentwindow
        self.setupUi(self)
        self.funcs = []


    def saveContourData(self, writer, images, labels):    
        self.funcs = []
        if self.parent.TrackingModeEnabled:
            self.saveTrackResults(writer, images, labels)
        else:
            self.saveSegmentationResults(writer, images, labels)

    def saveSegmentationResults(self,writer, images, labels):
        header = ['image name'] + ['frame number'] + ['object number'] + ['object type'] 

        header = self.addOptionalParameters(header)
                    
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.maxworker) as executor:
            rows = executor.map(self.getSingleContourLabelFromList, zip(images,labels))
        
        for r in rows:
             writer.writerows(r) 


    def addOptionalParameters(self, header):
        f1 = lambda x, _: x.getSize()
        f2 = lambda x, _: x.getSize() /((self.parent.canvas.scale_pixel_per_mm/1000)**2)
        f3 = lambda x, _: x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor) 
        f4 = lambda x, _: x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor)/(self.parent.canvas.scale_pixel_per_mm/1000)
        f5 = lambda x, _: x.getSize()/max(1,x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor))
        f6 = lambda x, _: x.getPerimeter()
        f7 = lambda x, _: x.getPerimeter()/(self.parent.canvas.scale_pixel_per_mm/1000)
        f8 = lambda x, y: getContourMinIntensity(y,x)
        f9 = lambda x, y: getContourMeanIntensity(y,x)
        f10 = lambda x, y: getContourMaxIntensity(y,x)

        if self.parentwindow.CBSize.isChecked():
            header += ['size in microns\u00b2']
            self.funcs += (f2,)
        else:
            header += ['size']
            self.funcs += (f1,)
        if self.CBSkeleton.isChecked():
            if self.parentwindow.CBSize.isChecked():
                header += ['length in microns']
                self.funcs += (f4,)
            else:
                header += ['length in pixel']
                self.funcs += (f3,)
            header += ['fatness (area / length)']
            self.funcs += (f5,)
        if self.CBPerimeter.isChecked(): 
            if self.parentwindow.CBSize.isChecked():
                header += ['perimeter in microns']
                self.funcs += (f7,)
            else:
                header += ['perimeter']
                self.funcs += (f6,)
        if self.CBMin.isChecked():
            header += ['minimum pixel intensity']
            self.funcs += (f8,)
        if self.CBMean.isChecked():
            header += ['mean pixel intensity']
            self.funcs += (f9,)
        if self.CBMax.isChecked():
            header += ['max pixel intensity']
            self.funcs += (f10,)

        return header

    def LoadImageIfNecessary(self, name, frame):
        if self.CBMin.isChecked() or self.CBMean.isChecked() or self.CBMax.isChecked(): 
            imagefile = ImageFile(self.parent.files.convertIfStackPath(name))
            image = imagefile.getImage(frame)
        else: 
            image = None
        return image

    def getFrameNumber(self, label):
        if self.parent.files.isStackLabel(label):
            frame = self.parent.files.getFrameNumber(label) + 1
        else:
            frame = 1
        return frame
        
    def getSingleContourLabelFromList(self, im_labels):

        name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(im_labels[0]),withfileextension=True)
        contours = loadContours(self.parent.files.convertIfStackPath(im_labels[1]))
        contours = [x for x in contours if x.isValid(self.parent.canvas.minContourSize)]
        frame = self.getFrameNumber(im_labels[1])
        x = 0
        rows = []
        for i,c in enumerate(contours):
            x += 1
            row = [name] + [frame] 
            row += [c.objectNumber] if c.objectNumber >= 0 else [i+1]
            row += [c.classlabel]
            image = self.LoadImageIfNecessary(im_labels[0],frame-1)
            row += [f(c,image) for f in self.funcs]
            rows.append(row)
        self.parent.emitProgress()
        return rows

    def saveTrackResults(self, writer, images, labels):
        header = ['object number'] + ['time point'] +  ['image name'] + ['frame number'] + ['object type'] + ['position'] 
        f1 = lambda x, _: x.classlabel
        f2 = lambda x, _: x.getCenter()
        self.funcs = (f1,f2)
        header = self.addOptionalParameters(header)

        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
        trackres = self.getTrackResults(images, labels)
        for r in trackres:
            writer.writerow(r) 
        
    def getTRackresults(self, images,labels):
        contours = []
        contours = [{} for x in range(self.parent.tracking.timepoints)]

        for i,l in zip(images,labels):
            name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(i),withfileextension=True)
            contour = loadContours(self.parent.files.convertIfStackPath(l))
            if self.parent.tracking.stackMode and self.parent.files.isStackLabel(l):
                tp = self.parent.files.getFrameNumber(l)
            else:
                tp = self.parent.tracking.getTimePointFromImageName(self.parent.files.convertIfStackPath(i))
            frame = self.getFrameNumber(l)

            image = self.LoadImageIfNecessary(i,frame-1)
            contours[tp] = {x.objectNumber: [name] + [frame] + [f(x,image) for f in self.funcs] for x in contour if x.isValid(self.parent.canvas.minContourSize)}
            self.parent.emitProgress()
        rows = []
        for n in self.parent.tracking.getNumbersOfTrackedObjects():
            for t in self.parent.tracking.getObjectOccurences(n):
                try:
                    row = [n] + [t+1] + contours[t][n]
                    rows.append(row)
                except:
                    pass
        return rows
