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
        Form.setWindowTitle('Results') 
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
        

        
        self.CBSize = QCheckBox("Export scaled size",self.centralWidget)
        self.CBSize.setChecked(True)
        self.CBSize.setToolTip('Use scale to calculate size from pixel values')
        self.vlayout.addWidget(self.CBSize)
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.BSetScale = DCDButton(self.centralWidget)
        self.BSetScale.setToolTip('Measure scale with known distance')
        self.BSetScale.setIcon(QIcon('icons/measure.png'))
        self.BSetScale.setMaximumSize(28, 28)
        self.hlayout.addWidget(self.BSetScale)
        
        self.LEScale = QLineEdit(self.centralWidget) 
        self.LEScale.setMaximumSize(100, 28)
        self.LEScale.setStyleSheet("border: None")
        self.LEScale.setStyleSheet('font:bold')
        self.LEScale.setToolTip('Scale in pixel per mm')
        self.LEScale.setAlignment(Qt.AlignRight)
        self.LEScale.setValidator(QIntValidator(1,999999999))
        self.hlayout.addWidget(self.LEScale)
        self.LScale = QLabel(self.centralWidget)
        self.LScale.setText('  pixel per mm')
        self.hlayout.addWidget(self.LScale)
        self.vlayout.addItem(self.hlayout)
        
        self.BExportMasks = DCDButton(self.centralWidget, 'Export Masks')
        self.BExportMasks.setToolTip('Export contour masks of all images in prediction folder')
        self.BExportMasks.setIcon(QIcon('icons/savemodel.png'))
        self.vlayout.addWidget(self.BExportMasks)
        
        self.BExport = DCDButton(self.centralWidget, 'Export as csv')
        self.BExport.setToolTip('Export data from all images in prediction folder')
        self.BExport.setIcon(QIcon('icons/savemodel.png'))
        self.BExport.setFlat(True)
        self.vlayout.addWidget(self.BExport)
        


class ResultsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ResultsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.BExportMasks.clicked.connect(self.exportAllMasks)
        self.BExport.clicked.connect(self.saveResults)
        self.CBSize.stateChanged.connect(self.EnableSetScale)
        self.BSetScale.clicked.connect(self.setScaleTool)
        self.LEScale.textEdited.connect(self.setScale)
        self.LEScale.setText('1')
        
    def exportMask(self):
        # currently unused
        image = self.parent.getCurrentImage()
        label = self.parent.dl.Mode.LoadLabel(self.parent.files.CurrentLabelPath(), image.shape[0], image.shape[1])
        filename = QFileDialog.getSaveFileName(self, "Save Mask To File", '', "Tiff (*.tif)")[0]
        if filename.strip():
            with self.parent.wait_cursor():
                cv2.imwrite(filename, label)
                
    def exportAllMasks(self):
        with self.parent.wait_cursor():
            images, labels = getAllImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath)
            folder = self.parent.files.testImagespath + os.path.sep + "exported_masks"

            if not images or not labels:
                self.parent.PopupWarning('No predicted masks')
                return
            image = None
            for i, l in zip(images,labels):
                if isinstance(l, tuple):
                    l = l[0]
                    if not image or image.path != i[0]:                
                        image = ImageFile(i[0])
                    label = self.parent.dl.Mode.LoadLabel(l, image.height(), image.width())
                    name = self.parent.files.getFilenameFromPath(l)
                    f = self.parent.files.extendLabelPathByFolder(folder, self.parent.files.getFilenameFromPath(i[0]))
                    self.parent.files.ensureFolder(f)
                    cv2.imwrite(os.path.join(f,name)+'.tif', label)
                else:
                    image = ImageFile(i)
                    label = self.parent.dl.Mode.LoadLabel(l, image.height(), image.width())
                    name = self.parent.files.getFilenameFromPath(l)
                    self.parent.files.ensureFolder(folder)
                    cv2.imwrite(os.path.join(folder,name)+'.tif', label)
            self.hide()
        
    def setScale(self, text):
        # qtvalidator does not have a status request so we need to validate again
        if self.LEScale.validator().validate(text,0)[0] == QValidator.Acceptable:
            self.parent.canvas.setScale(int(text))


    def setScaleTool(self):
        self.parent.canvas.setnewTool(canvasTool.scale.name)
        

    def EnableSetScale(self):
        if self.CBSize.isChecked():
            self.LEScale.setEnabled(True)
            self.BSetScale.setEnabled(True)
            self.LScale.setStyleSheet("color: black")
        else:
            self.LEScale.setEnabled(False)
            self.BSetScale.setEnabled(False)
            self.LScale.setStyleSheet("color: gray")

    def saveResults(self):
        filename = QFileDialog.getSaveFileName(self, "Save Results", '', "Comma Separated File (*.csv)")[0]
        if not filename.strip():
            return
        
        with self.parent.wait_cursor():
            _, labels = getAllImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath)
            if labels is None:
                self.parent.PopupWarning('No predicted files')
                return
            # try:
            if True:
                with open(filename, 'w', newline='') as csvfile:
                    csvWriter = csv.writer(csvfile, delimiter = ';')
                        
                    if self.parent.LearningMode() == dlMode.Segmentation:
                        self.saveContourData(csvWriter, labels)
                    elif self.parent.LearningMode() == dlMode.Object_Counting:
                        self.saveObjects(csvWriter, labels)
                    else:
                        self.parent.PopupWarning('Not supported detection mode')
                    
                    
                    self.parent.writeStatus('results saved')
                
            # except: 
            #     self.parent.PopupWarning('Cannot write file (already open?)')
            #     return
            
            self.hide()
            
    def saveObjects(self, writer, labels):
        header = ['image name'] + ['number of objects'] + ['object type']
        writer.writerow(header)
        num = len(labels)
        self.parent.initProgress(num)
        
        for label in labels:
            if isinstance(label, tuple):
                label = label[0]
                
            name = os.path.splitext(os.path.basename(label))[0]
            points,_ = loadPoints(label)
            pointlabels = [x.classlabel for x in points]
            for i in range(1,max(pointlabels)+1):
                row = [name] + [pointlabels.count(i)] + [i]
                writer.writerow(row) 
        
            
    def saveContourData(self, writer, labels):    
        header = ['image name'] + ['object number'] + ['object type'] + ['size']
        if self.CBSize.isChecked():
            header += ['size in microns\u00b2']
        if self.parent.canvas.drawSkeleton:
            header += ['length in pixel']
            if self.CBSize.isChecked():
                header += ['length in microns']
            header += ['fatness (area / length)']
                        
        writer.writerow(header)
        num = len(labels)
        self.parent.initProgress(num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.maxworker) as executor:
            rows = executor.map(self.getSingleContourLabelFromList, labels)
                    
        for r in rows:
            writer.writerows(r) 
        self.parent.ProgressFinished()
        
    def getSingleContourLabelFromList(self, contourname):
        if isinstance(contourname, tuple):
            contourname = contourname[0]
        contours = loadContours(contourname)
        name = os.path.splitext(os.path.basename(contourname))[0]
        x = 0
        rows = []
        for c in contours:
            x += 1
            row = [name] + [x] + [c.classlabel] + [c.getSize()]
            if self.CBSize.isChecked():
                row += ['%.2f'%(c.getSize()/((self.parent.canvas.scale_pixel_per_mm/1000)**2))]
            if self.parent.canvas.drawSkeleton:   
                length = c.getSkeletonLength()   
                row += ['%.0f'%length]
                if self.CBSize.isChecked():
                    row += ['%.2f'%(length/(self.parent.canvas.scale_pixel_per_mm/1000))]
                if length > 0:
                    row += ['%.2f'%(c.getSize()/length)]
            rows.append(row)
        self.parent.addProgress()
        return rows
       
            
        
                   
        
        

            

            
