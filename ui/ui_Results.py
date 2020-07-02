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
from dl.dl_labels import LoadLabel
from utils.Contour import loadContours
from utils.utility_functions import getAllImageLabelPairPaths
from ui.Tools import canvasTool
from ui.style import styleForm
from ui.ui_utils import DCDButton
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
        self.CBSize.clicked.connect(self.EnableSetScale)
        self.BSetScale.clicked.connect(self.setScaleTool)
        self.LEScale.textEdited.connect(self.setScale)
        self.LEScale.setText('1')
        
    def exportMask(self):
        # currently unused
        image = self.parent.getCurrentImage()
        label = LoadLabel(self.parent.CurrentLabelFullName(), image.shape[0], image.shape[1])
        filename = QFileDialog.getSaveFileName(self, "Save Mask To File", '', "Tiff (*.tif)")[0]
        if filename.strip():
            with self.parent.wait_cursor():
                cv2.imwrite(filename, label)
                
    def exportAllMasks(self):
        with self.parent.wait_cursor():
            images, labels = getAllImageLabelPairPaths(self.parent.testImagespath,self.parent.testImageLabelspath)
            folder = self.parent.testImagespath + "/exported_masks/"
            self.parent.ensureFolder(folder)
            for i, l in zip(images,labels):
                image = cv2.imread(i)
                label = LoadLabel(l, image.shape[0], image.shape[1])
                name = self.parent.getFilenameFromPath(l)
                cv2.imwrite(folder+name+'.tif', label)
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
            _, labels = getAllImageLabelPairPaths(self.parent.testImagespath,self.parent.testImageLabelspath)
            if labels is None:
                self.parent.PopupWarning('No predicted files')
                return
       
        
            # try:
            if True:
                with open(filename, 'w', newline='') as csvfile:
                    csvWriter = csv.writer(csvfile, delimiter = ';')
                    header = ['image name'] + ['object number'] + ['object type'] + ['size']
                    if self.CBSize.isChecked():
                        header += ['size in microns\u00b2']
                    if self.parent.canvas.drawSkeleton:
                        header += ['length in pixel']
                        if self.CBSize.isChecked():
                            header += ['length in microns']
                        header += ['fatness (area / length)']
                        
                    csvWriter.writerow(header)
                    num = len(labels)
                    self.parent.initProgress(num)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.parent.maxworker) as executor:
                        rows = executor.map(self.getSingleLabelFromList, labels)
                    
                    for r in rows:
                        csvWriter.writerows(r) 
                    self.parent.ProgressFinished()
                    self.parent.writeStatus('results saved')
                
            # except: 
            #     self.parent.PopupWarning('Cannot write file (already open?)')
            #     return
            
            self.hide()
        
    def getSingleLabelFromList(self, contourname):
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
       
            
        
                   
        
        

            

            
