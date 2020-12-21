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
from ui.segmentation.ui_SegResults import SegmentationResultsWindow
from ui.objectdetection.ui_ODResults import ObjectDetectionResultsWindow
from dl.data.labels import getMatchingImageLabelPairPaths, getMatchingImageLabelPairsRecursive
from utils.shapes.Contour import loadContours
from utils.shapes.Point import loadPoints
from ui.Tools import canvasTool
from ui.style import styleForm
from ui.ui_utils import DCDButton, saveFile
from utils.Image import ImageFile
from dl.method.mode import dlMode
from utils.workerthread import WorkerThread
import cv2
import threading


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Results') 
        width = 250
        height= 130
        Form.setFixedSize(width, height)
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        self.CBSize = QCheckBox("Use scaled size",self.centralWidget)
        self.CBSize.setObjectName('ExportScaledSize')
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
        self.LEScale.setObjectName('Scale')
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
        
        self.BExportSettings = DCDButton(self.centralWidget, 'Settings')
        self.BExportSettings.setToolTip('Open export settings')
        self.BExportSettings.setIcon(QIcon('icons/savemodel.png'))
        self.vlayout.addWidget(self.BExportSettings)
        
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
        self.BExportSettings.clicked.connect(self.showExportSettings)
        self.CBSize.stateChanged.connect(self.EnableSetScale)
        self.BSetScale.clicked.connect(self.setScaleTool)
        self.LEScale.textEdited.connect(self.setScale)
        self.LEScale.setText('1')
        
        self.odResults = ObjectDetectionResultsWindow(self.parent,self)
        self.segResults = SegmentationResultsWindow(self.parent,self)
        
    def showExportSettings(self):
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.segResults.show()
        elif self.parent.LearningMode() == dlMode.Object_Counting:
            self.odResults.show()
            
    def hide(self):
        self.closeWindows()
        super(ResultsWindow, self).hide()
        
    def closeEvent(self, event):       
        self.closeWindows()
        super(ResultsWindow, self).closeEvent(event)
            
    def ModeChanged(self):
        self.closeWindows()
        
    
    def closeWindows(self):
        self.segResults.hide()
        self.odResults.hide()

        
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
            images, labels = getMatchingImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath)
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
        # filename = QFileDialog.getSaveFileName(self, "Save Results", '', "Comma Separated File (*.csv)")[0]
        filename = saveFile("Save Results","Comma Separated File (*.csv)", 'csv') 
        if not filename:
            return
        
        thread = WorkerThread(work=self.saving, args=filename, callback_start=self.parent.emitBlockWindow,callback_end=self.parent.emitUnBlockWindow)
        # thread.daemon = True
        thread.start()


        
        
    def saving(self, filename):
            #_, labels = getMatchingImageLabelPairsRecursive(self.parent.files.testImagespath,self.parent.dl.LabelFolderName)
            _, labels = getMatchingImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath)
            if labels is None or not labels:
                self.parent.emitPopup('No predicted files')
                return
            try:
                with open(filename, 'w', newline='') as csvfile:
                    csvWriter = csv.writer(csvfile, delimiter = ';')
                        
                    if self.parent.LearningMode() == dlMode.Segmentation:
                        self.segResults.saveContourData(csvWriter, labels)
                    elif self.parent.LearningMode() == dlMode.Object_Counting:
                        self.odResults.saveObjects(csvWriter, labels)
                    else:
                        self.parent.emitPopup('Not supported detection mode')

                    self.parent.emitProgressFinished()
                    self.parent.emitStatus('results saved')

            except: 
                self.parent.emitPopup('Cannot write file (already open?)')
                return

        
        

            

            
