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
        
        
        self.CBSkeleton = QCheckBox("Export Skeleton",self.centralWidget)
        self.CBSkeleton.setObjectName('ExportSkeleton')
        self.CBSkeleton.setChecked(True)
        self.CBSkeleton.setToolTip('Check to export skeleton length')

        self.vlayout.addWidget(self.CBSkeleton)
        


class SegmentationResultsWindow(QMainWindow, Window):
    def __init__(self, parent, parentwindow):
        super(SegmentationResultsWindow, self).__init__(parent)
        self.parent = parent
        self.parentwindow = parentwindow
        self.setupUi(self)


    def saveContourData(self, writer, images, labels):    
        if self.parent.TrackingModeEnabled:
            self.saveTrackResults(writer, images, labels)
        else:
            self.saveSegmenatationResults(writer, images, labels)

    def saveSegmenatationResults(self,writer, images, labels):
         header = ['image name'] + ['frame number'] + ['object number'] + ['object type'] + ['size']
         if self.parentwindow.CBSize.isChecked():
             header += ['size in microns\u00b2']
         if self.CBSkeleton.isChecked():
             header += ['length in pixel']
             if self.parentwindow.CBSize.isChecked():
                 header += ['length in microns']
             header += ['fatness (area / length)']
                     
         writer.writerow(header)
         num = len(labels)
         self.parent.emitinitProgress(num)
         with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.maxworker) as executor:
             rows = executor.map(self.getSingleContourLabelFromList, zip(images,labels))
        
         for r in rows:
             writer.writerows(r) 
        
    def getSingleContourLabelFromList(self, im_labels):

        name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(im_labels[0]),withfileextension=True)
        contours = loadContours(self.parent.files.convertIfStackPath(im_labels[1]))
        contours = [x for x in contours if x.isValid(self.parent.canvas.minContourSize)]
        if self.parent.files.isStackLabel(im_labels[1]):
            frame = self.parent.files.getFrameNumber(im_labels[1]) + 1
        else:
            frame = 1
        x = 0
        rows = []
        for c in contours:
            x += 1
            row = [name] + [frame] + [x] + [c.classlabel] + [c.getSize()]
            if self.parentwindow.CBSize.isChecked():
                row += ['%.2f'%(c.getSize()/((self.parent.canvas.scale_pixel_per_mm/1000)**2))]
            if self.CBSkeleton.isChecked():   
                length = c.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor)   
                row += ['%.0f'%length]
                if self.parentwindow.CBSize.isChecked():
                    row += ['%.2f'%(length/(self.parent.canvas.scale_pixel_per_mm/1000))]
                row += ['%.2f'%(c.getSize()/max(1,length))]
            rows.append(row)
        self.parent.emitProgress()
        return rows

    def saveTrackResults(self, writer, images, labels):
        header = ['object number'] + ['time point'] +  ['image name'] + ['frame number'] + ['object type'] + ['position'] + ['size']
        if self.parentwindow.CBSize.isChecked():
            header += ['size in microns\u00b2']
        if self.CBSkeleton.isChecked():
            header += ['length in pixel']
            if self.parentwindow.CBSize.isChecked():
                header += ['length in microns']
            header += ['fatness (area / length)']
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
        trackres = self.getTRackResults(images, labels)
        for r in trackres:
            writer.writerow(r) 
        
    def getTRackResults(self, images,labels):
        contours = []

        f1 = lambda x: x.classlabel
        f2 = lambda x: x.getCenter()
        f3 = lambda x: x.getSize()
        f4 = lambda x: x.getSize() /((self.parent.canvas.scale_pixel_per_mm/1000)**2)
        f5 = lambda x: x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor) 
        f6 = lambda x: x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor)/(self.parent.canvas.scale_pixel_per_mm/1000)
        f7 = lambda x: x.getSize()/max(1,x.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor))
        # f8 = lambda x, y = image: getContourMinIntensity(y,x)
        # f9 = lambda x, y = image: getContourMeanIntensity(y,x)
        # f10 = lambda x, y = image: getContourMaxIntensity(y,x)
        # f11 = lambda x: x.getPerimeter()

        cnt_data = (f1,f2,f3)
        if self.parentwindow.CBSize.isChecked():
            cnt_data += (f4,)
        if self.CBSkeleton.isChecked(): 
            cnt_data += (f5,)
            if self.parentwindow.CBSize.isChecked():
                cnt_data += (f6,)
            cnt_data += (f7,)

        
        
        # add additional requirements here
        contours = [{} for x in range(self.parent.tracking.timepoints)]

        for i,l in zip(images,labels):
            name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(i),withfileextension=True)
            contour = loadContours(self.parent.files.convertIfStackPath(l))
            if self.parent.tracking.stackMode:
                tp = self.parent.files.getFrameNumber(l)
            else:
                tp = self.parent.tracking.getTimePointFromImageName(self.parent.files.convertIfStackPath(i))
            if self.parent.files.isStackLabel(l):
                frame = self.parent.files.getFrameNumber(l) + 1
            else:
                frame = 1
            contours[tp] = {x.objectNumber: [name] + [frame] + [f(x) for f in cnt_data] for x in contour if x.isValid(self.parent.canvas.minContourSize)}
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
