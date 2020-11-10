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
from dl.data.labels import getAllImageLabelPairPaths
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

    def saveContourData(self, writer, labels):    
        header = ['image name'] + ['object number'] + ['object type'] + ['size']
        if self.parentwindow.CBSize.isChecked():
            header += ['size in microns\u00b2']
        if self.CBSkeleton.isChecked():
            header += ['length in pixel']
            if self.parentwindow.CBSize.isChecked():
                header += ['length in microns']
            header += ['fatness (area / length)']
                        
        writer.writerow(header)
        num = len(labels)
        self.parent.initProgress(num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.maxworker) as executor:
            rows = executor.map(self.getSingleContourLabelFromList, labels)
                    
        for r in rows:
            writer.writerows(r) 
        
    def getSingleContourLabelFromList(self, contourname):
        if isinstance(contourname, tuple):
            name = os.path.splitext(os.path.basename(contourname[0]))[0][:-3] + str(int(contourname[1])+1)
            contourname = contourname[0]
        else:
            name = os.path.splitext(os.path.basename(contourname))[0]
        
        contours = loadContours(contourname)
        contours = [x for x in contours if x.isValid(self.parent.canvas.minContourSize)]
        
        x = 0
        rows = []
        for c in contours:
            x += 1
            row = [name] + [x] + [c.classlabel] + [c.getSize()]
            if self.parentwindow.CBSize.isChecked():
                row += ['%.2f'%(c.getSize()/((self.parent.canvas.scale_pixel_per_mm/1000)**2))]
            if self.CBSkeleton.isChecked():   
                length = c.getSkeletonLength(self.parent.canvas.skeletonsmoothingfactor)   
                row += ['%.0f'%length]
                if self.parentwindow.CBSize.isChecked():
                    row += ['%.2f'%(length/(self.parent.canvas.scale_pixel_per_mm/1000))]
                if length > 0:
                    row += ['%.2f'%(c.getSize()/length)]
            rows.append(row)
        self.parent.Progress.emit()
        return rows
       
            
        
                   
        
        

            

            
