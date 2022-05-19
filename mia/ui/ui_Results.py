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
from ui.classification.ui_ClassResults import ClassificationResultsWindow
from dl.data.labels import getMatchingImageLabelPairPaths, getMatchingImageLabelPairsRecursive
from utils.shapes.Contour import loadContours
from utils.shapes.Point import loadPoints
from ui.Tools import canvasTool
from ui.style import styleForm
from ui.ui_utils import DCDButton, saveFile, openFolder
from utils.Image import ImageFile
from dl.method.mode import dlMode
from utils.workerthread import WorkerThread
import cv2
import threading
import sys

class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Results') 

        styleForm(Form)
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        self.CBSize = QCheckBox("Use scaled size",self.centralWidget)
        self.CBSize.setObjectName('ExportScaledSize')
        self.CBSize.setChecked(True)
        self.CBSize.setToolTip('Use scale to calculate size from pixel values')
        self.vlayout.addWidget(self.CBSize)
                
        self.hlayout = QHBoxLayout()
        self.BSetScale = DCDButton(self.centralWidget)
        self.BSetScale.setToolTip('Measure scale with known distance')
        self.BSetScale.setIcon(QIcon('icons/measure.png'))
        self.BSetScale.setMaximumSize(28, 28)
        self.hlayout.addWidget(self.BSetScale)
        
        self.LEScale = QLineEdit(self.centralWidget) 
        self.LEScale.setMaximumSize(100, 28)
        self.LEScale.setObjectName('Scale')
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
        self.BExportSettings.setIcon(QIcon('icons/settings.png'))
        self.vlayout.addWidget(self.BExportSettings)
        
        hlayout = QHBoxLayout()
        self.BExportMasks = DCDButton(self.centralWidget, 'Export all Masks')
        self.BExportMasks.setToolTip('Export contour masks of all images in prediction folder')
        self.BExportMasks.setIcon(QIcon('icons/exportallmasks.png'))
        self.BExportMask = DCDButton(self.centralWidget, 'Export Mask')
        self.BExportMask.setToolTip('Export contour mask of current image')
        self.BExportMask.setIcon(QIcon('icons/exportmask.png'))
        hlayout.addWidget(self.BExportMask)
        hlayout.addWidget(self.BExportMasks)
        self.vlayout.addLayout(hlayout)
        
        hlayout2 = QHBoxLayout()
        self.BExport = DCDButton(self.centralWidget, 'Save')
        self.BExport.setToolTip('save data from current image')
        self.BExport.setIcon(QIcon('icons/savemodel.png'))
        self.BExport.setFlat(True)
        self.BExportAll = DCDButton(self.centralWidget, 'Save all')
        self.BExportAll.setToolTip('Save data from all images in prediction folder')
        self.BExportAll.setIcon(QIcon('icons/saveall.png'))
        self.BExportAll.setFlat(True)

        hlayout2.addWidget(self.BExport)
        hlayout2.addWidget(self.BExportAll)
        self.vlayout.addLayout(hlayout2)

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())
        


class ResultsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ResultsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.BExportMasks.clicked.connect(self.exportAllMasks)
        self.BExportMask.clicked.connect(self.exportMask)
        self.BExportAll.clicked.connect(self.saveResults)
        self.BExport.clicked.connect(self.saveCurrent)
        self.BExportSettings.clicked.connect(self.showExportSettings)
        self.CBSize.stateChanged.connect(self.EnableSetScale)
        self.BSetScale.clicked.connect(self.setScaleTool)
        self.LEScale.textEdited.connect(self.setScale)
        self.LEScale.setText('1')
        
        self.odResults = ObjectDetectionResultsWindow(self.parent,self)
        self.segResults = SegmentationResultsWindow(self.parent,self)
        self.classResults = ClassificationResultsWindow(self.parent,self)
        
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
        self.classResults.hide()
        
    def exportMask(self):
        if not os.path.exists(self.parent.files.CurrentLabelPath()) or self.parent.getCurrentImage() is None:
            return
        # currently unused
        image = self.parent.getCurrentImage()
        label = self.parent.dl.Mode.LoadLabel(self.parent.files.CurrentLabelPath(), image.shape[0], image.shape[1], rmBackground=True)
        
        #filename = QFileDialog.getSaveFileName(self, "Save Mask To File", '', "Tiff (*.tif)")[0]
        filename = saveFile("Save Mask To File", "Tiff (*.tif)", 'tif')
        if filename.strip():
            with self.parent.wait_cursor():
                cv2.imwrite(filename, label)
                
    def exportAllMasks(self):
        folder = openFolder(("Select Target Folder"))
        if not folder:
            return
        with self.parent.wait_cursor():
            images, labels = getMatchingImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath, self.parent.dl.data.stacklabels)

            if not images or not labels:
                self.parent.PopupWarning('No predicted masks')
                return
            

            image = None
            for i, l in zip(images,labels):
                if isinstance(l, tuple):
                    l = l[0]
                    if not image or image.path != i[0]:                
                        image = ImageFile(i[0])
                    label = self.parent.dl.Mode.LoadLabel(l, image.height(), image.width(), rmBackground=True)
                    name = self.parent.files.getFilenameFromPath(l)
                    f = self.parent.files.extendLabelPathByFolder(folder, self.parent.files.getFilenameFromPath(i[0]))
                    self.parent.files.ensureFolder(f)
                    cv2.imwrite(os.path.join(f,name)+'.tif', label)
                else:
                    image = ImageFile(i)
                    label = self.parent.dl.Mode.LoadLabel(l, image.height(), image.width(), rmBackground=True)
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

    def saveCurrent(self):
        filename = saveFile("Save Results","Comma Separated File (*.csv)", 'csv') 
        if not filename:
            return
        label = self.parent.files.CurrentLabelPath()
        image = self.parent.files.CurrentImagePath()
        if not os.path.exists(label):
            self.parent.emitPopup('No prediction')
            return
        self.saving([image], [label], filename)


    def saveResults(self):
        # filename = QFileDialog.getSaveFileName(self, "Save Results", '', "Comma Separated File (*.csv)")[0]
        filename = saveFile("Save Results", "Comma Separated File (*.csv)", 'csv') 
        if not filename:
            return
        
        thread = WorkerThread(work=self.saveAll, args=filename, callback_start=self.parent.emitBlockWindow,callback_end=self.parent.emitUnBlockWindow)
        # thread.daemon = True
        thread.start()

    def saveAll(self,filename):
        images, labels = getMatchingImageLabelPairPaths(self.parent.files.testImagespath,self.parent.files.testImageLabelspath, self.parent.dl.data.stacklabels)

        if labels is None or not labels:
            self.parent.emitPopup('No predicted files')
            return
        self.saving(images,labels, filename)

    def saving(self, images, labels, filename):
         try:
            with open(filename, 'w', newline='') as csvfile:
                csvWriter = csv.writer(csvfile, delimiter = ';')
                    
                if self.parent.LearningMode() == dlMode.Segmentation:
                    self.segResults.saveContourData(csvWriter, images, labels)
                elif self.parent.LearningMode() == dlMode.Object_Counting:
                    self.odResults.saveObjects(csvWriter, images, labels)
                elif self.parent.LearningMode() == dlMode.Classification:
                    self.classResults.saveClasses(csvWriter, images, labels)
                else:
                    self.parent.emitPopup('Detection mode not supported')

                self.parent.emitProgressFinished()
                self.parent.emitStatus('results saved')

         except PermissionError: 
             self.parent.emitPopup('Cannot write file (already open?)')
             return


