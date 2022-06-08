# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:28:08 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from tensorflow import errors

import io
import traceback
import glob
import os
import cv2 
import multiprocessing
import threading
import concurrent.futures
import time
import webbrowser
from contextlib import contextmanager


import dl.DeepLearning as DeepLearning
from dl.method.mode import dlMode
from dl.data.labels import dlStackLabels
from ui.UI import MainWindow
from ui.Tools import canvasTool, canvasToolButton
from ui.ui_Canvas import Canvas

from ui.segmentation.ui_SegPostProcessing import SegmentationPostProcessingWindow
from ui.objectdetection.ui_ODPostProcessing import ObjectDetectionPostProcessingWindow

from ui.ui_Results import ResultsWindow

from ui.ui_Training import TrainingWindow
from ui.ui_Settings import SettingsWindow
from ui.ui_Log import LogWindow

from ui.ui_TrainPlot import TrainPlotWindow
from ui.settings import Settings

import utils.shapes.Contour as Contour
import utils.shapes.Point as Point
from utils.Image import ImageFile, supportedImageFormats
from utils.Observer import QtObserver
from utils.FilesAndFolders import FilesAndFolders
from utils.postprocessing.tracking import ObjectTracking


from utils.workerthread import WorkerThread
from ui.ui_utils import DCDButton, openFolder, saveFile, loadFile

import pickle

import numpy as np

LOG_FILENAME = 'log/logfile.log'
SETTINGS_FILENAME = 'settings/user_settings.json'



class MIA_UI(QMainWindow, MainWindow):
    # qt elements may only be changed from main thread, so we create signals here
    # to call from other threads
    initProgress = pyqtSignal(int)
    Progress = pyqtSignal() 
    ProgressFinished = pyqtSignal()
    Block = pyqtSignal() 
    UnBlock = pyqtSignal()
    Popup = pyqtSignal(str)
    StatusMessage = pyqtSignal(str)
    
    def __init__(self):
        super(MIA_UI, self).__init__()
        self.initialized = False
        self.setupUi(self)
        self.setCallbacks()
        self.show()

        self.progresstick = 10
        self.progress = 0
        self.progresslock = threading.Lock()
        self.processlock = threading.Lock()
        self.maxworker = 1
        self.lock = threading.Lock()
        
        self.files = FilesAndFolders(self)
        self.files.ensureFolder('log/')
        self.files.ensureFolder('settings/')
        
        self.currentImageFile = None
        self.keepImageSettings = False

        self.setFocusPolicy(Qt.NoFocus)
        width = self.canvas.geometry().width()
        height = self.canvas.geometry().height() 
        
        
        self.predictionRate = 0
        self.saveload_modelweights_only = True
        
        self.dl = DeepLearning.DeepLearning()
        self.tracking = ObjectTracking(self.dl, self.files)
        # init observer
        self.dlobserver = QtObserver()
        self.dl.attachObserver(self.dlobserver)
        self.dlobserver.signals.progress.connect(self.dlProgress)
        self.dlobserver.signals.epoch_end.connect(self.dlEpoch)
        self.dlobserver.signals.finished.connect(self.dlFinished)
        self.dlobserver.signals.result.connect(self.dlResult)
        self.dlobserver.signals.error.connect(self.dlError)
        self.dlobserver.signals.start.connect(self.dlStarted)
        
        
        self.initProgress.connect(self.initializeProgress)
        self.Progress.connect(self.addProgress)
        self.ProgressFinished.connect(self.finishProgress)
        self.Block.connect(self.blockWindow)
        self.UnBlock.connect(self.unblockWindow)
        self.Popup.connect(self.PopupWarning)
        self.StatusMessage.connect(self.writeStatus)
        
        # init canvas
        self.hlayout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas = Canvas(self)
        self.canvas.setMouseTracking(True)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFixedSize(QSize(width, height))
        self.canvas.reset()
        self.hlayout.addWidget(self.canvas)
        self.hlayout.addWidget(self.StackSettings)
        horizontalspacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hlayout.addItem(horizontalspacer)
        
        self.classList.setClass(0)
        self.changeLearningMode(self.LearningMode().value)
        self.CBLearningMode.setCurrentIndex (self.LearningMode().value)
        self.setWorkers(multiprocessing.cpu_count() // 2)

        ### windows - should this be moved to UI ?
        self.segpostprocessing_form = SegmentationPostProcessingWindow(self)
        self.results_form = ResultsWindow(self)
        self.odpostprocessing_form = ObjectDetectionPostProcessingWindow(self)
        self.training_form = TrainingWindow(self)
        self.settings_form = SettingsWindow(self)
        self.plotting_form = TrainPlotWindow(self)
        self.log_form = LogWindow(self)
        self.ClassListUpdated()
        self.classList.setClass(1)
        self.enableCopyStackLabels()
        
        self.initialized = True
        self.settings = Settings(self)
        self.settings.loadSettings(SETTINGS_FILENAME)
        self.defineShortcuts()


    def setCallbacks(self):
        self.Bclear.clicked.connect(self.clear)
        self.Bundo.clicked.connect(self.undo)
        self.BLoadTrainImages.clicked.connect(self.setTrainFolder)
        self.BLoadTestImages.clicked.connect(self.setTestFolder)
        self.BTestImageFolder.clicked.connect(self.switchToTestFolder)
        self.BTrainImageFolder.clicked.connect(self.switchToTrainFolder)
        self.Bnext.clicked.connect(self.nextImage)
        self.Bprev.clicked.connect(self.previousImage)
        self.SFrame.valueChanged.connect(self.FrameChanged)
        self.BCopyStackLabel.clicked.connect(self.copyStackLabels)
        
        self.Bdrag.clicked.connect(self.setCanvasMode)
        self.Bdraw.clicked.connect(self.setCanvasMode)
        self.Bassign.clicked.connect(self.setCanvasMode)
        self.Bextend.clicked.connect(self.setCanvasMode)
        self.Bdelete.clicked.connect(self.setCanvasMode)
        self.Bpoly.clicked.connect(self.setCanvasMode)
        self.Bassist.clicked.connect(self.setCanvasMode)
        
        self.BassignClass.clicked.connect(self.setCanvasMode)
        self.BsetClass.clicked.connect(self.setCanvasMode)
        self.BsetallClass.clicked.connect(self.setCanvasMode)
        
        self.BsetObject.clicked.connect(self.setCanvasMode)
        self.BshiftObject.clicked.connect(self.setCanvasMode)

        self.BSetObjectNumber.clicked.connect(self.setCanvasMode)
        self.BDeleteObject.clicked.connect(self.setCanvasMode)
        self.BChangeObjectColor.clicked.connect(self.setCanvasMode)

        
        self.Btrain.clicked.connect(self.showTrainingWindow)
        self.Bpredictall.clicked.connect(self.predictAllImages)
        self.Bpredict.clicked.connect(self.predictImage)
        self.Bloadmodel.clicked.connect(self.loadModel)
        self.Bsavemodel.clicked.connect(self.saveModel)
        self.Bautoseg.clicked.connect(self.autoSegment)
        self.CBSmartMode.stateChanged.connect(self.setSmartMode)
        self.CBInner.stateChanged.connect(self.updateImage)

        self.Bresetmodel.clicked.connect(self.resetModel)
        self.Bautodect.clicked.connect(self.autoPointDetection)
        
        
        self.Bresults.clicked.connect(self.showResultsWindow)
        self.BSettings.clicked.connect(self.showSettingsWindow)
        self.BPostProcessing.clicked.connect(self.showPostProcessingWindow)
        
        self.Baddclass.clicked.connect(self.addClass)
        self.Bdelclass.clicked.connect(self.removeLastClass)

        self.BTrack.clicked.connect(self.calcTracking)
        self.CBshowtrack.stateChanged.connect(self.updateImage)
        self.CBtrackcolor.stateChanged.connect(self.updateImage)
        

        self.CBLearningMode.currentIndexChanged.connect(self.changeLearningMode)
        self.CBExtend.clicked.connect(self.updateCursor)
        self.CBErase.clicked.connect(self.updateCursor)
        self.CBAddShape.clicked.connect(self.setCanvasFocus)
        self.CBDelShape.clicked.connect(self.setCanvasFocus)
        self.CBSliceShape.clicked.connect(self.setCanvasFocus)
        
        self.SBrightness.sliderReleased.connect(self.setBrightness)
        self.SContrast.sliderReleased.connect(self.setContrast)
        self.CBKeep.stateChanged.connect(self.keepImageSettingsChanged)

        self.TVFiles.clicked.connect(self.FileExplorerItemSelected)
      
        
        ## menu actions
        self.ASetTrainingFolder.triggered.connect(self.setTrainFolder)
        self.ASetTestFolder.triggered.connect(self.setTestFolder)
        self.ASaveSettings.triggered.connect(self.SaveSettings)
        self.ALoadSettings.triggered.connect(self.LoadSettings)
        self.ASaveDeepLearning.triggered.connect(self.saveDLClass)
        self.AOpenManual.triggered.connect(self.OpenManual)
        self.AOpenGithub.triggered.connect(self.OpenGithub)
        self.AOpenLog.triggered.connect(self.showLog)
        
        self.AExit.triggered.connect(self.close)
        self.ASettings.triggered.connect(self.showSettingsWindow)
        
        # monkey patching download request for model weights
        import dl.utils.dl_downloads as down
        down.callBackstart = lambda: self.ConfirmPopup('Do you want to download model weights')
        down.callBackfail = lambda: self.PopupWarning('Failed to download model weights')
        
    def defineShortcuts(self):
        self.sc_toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.sc_toggle.activated.connect(self.canvas.toggleDrag) 
        
        self.sc_prev = QShortcut(QKeySequence(Qt.Key_A), self)
        self.sc_prev.activated.connect(self.previousImage) 
        
        self.sc_next = QShortcut(QKeySequence(Qt.Key_D), self)
        self.sc_next.activated.connect(self.nextImage) 
        
        self.sc_prevframe = QShortcut(QKeySequence(Qt.Key_W), self)
        self.sc_prevframe.activated.connect(self.previousFrame) 
        
        self.sc_nextframe = QShortcut(QKeySequence(Qt.Key_S), self)
        self.sc_nextframe.activated.connect(self.nextFrame) 
        
        self.sc_class0 = QShortcut(48, self)
        self.sc_class0.activated.connect(self.setClassShortcut) 
        
        self.sc_class1 = QShortcut(49, self)
        self.sc_class1.activated.connect(self.setClassShortcut) 
        
        self.sc_class2 = QShortcut(50, self)
        self.sc_class2.activated.connect(self.setClassShortcut) 
        
        self.sc_class3 = QShortcut(51, self)
        self.sc_class3.activated.connect(self.setClassShortcut)
        
        self.sc_class4 = QShortcut(52, self)
        self.sc_class4.activated.connect(self.setClassShortcut) 
        
        self.sc_class5 = QShortcut(53, self)
        self.sc_class5.activated.connect(self.setClassShortcut) 
        
        self.sc_class6 = QShortcut(54, self)
        self.sc_class6.activated.connect(self.setClassShortcut) 
        
        self.sc_class7 = QShortcut(55, self)
        self.sc_class7.activated.connect(self.setClassShortcut)
        
        self.sc_class8 = QShortcut(56, self)
        self.sc_class8.activated.connect(self.setClassShortcut) 
        
        self.sc_class9 = QShortcut(57, self)
        self.sc_class9.activated.connect(self.setClassShortcut) 
        
        self.sc_undo = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_Z), self)
        self.sc_undo.activated.connect(self.undo) 

        self.sc_zoom_in = QShortcut(QKeySequence(Qt.Key_Plus), self)
        self.sc_zoom_in.activated.connect(self.canvas.zoomIn) 
        
        self.sc_zoom_out = QShortcut(QKeySequence(Qt.Key_Minus), self)
        self.sc_zoom_out.activated.connect(self.canvas.zoomOut) 
        
        self.sc_tool1 = QShortcut(QKeySequence(Qt.Key_F1), self)
        self.sc_tool1.activated.connect(self.setToolShortcut)
        
        self.sc_tool2 = QShortcut(QKeySequence(Qt.Key_F2), self)
        self.sc_tool2.activated.connect(self.setToolShortcut)
        
        self.sc_tool3 = QShortcut(QKeySequence(Qt.Key_F3), self)
        self.sc_tool3.activated.connect(self.setToolShortcut)
        
        self.sc_tool4 = QShortcut(QKeySequence(Qt.Key_F4), self)
        self.sc_tool4.activated.connect(self.setToolShortcut)
        
        self.sc_tool5 = QShortcut(QKeySequence(Qt.Key_F5), self)
        self.sc_tool5.activated.connect(self.setToolShortcut)
        
        self.sc_tool6 = QShortcut(QKeySequence(Qt.Key_F6), self)
        self.sc_tool6.activated.connect(self.setToolShortcut)

            
    def setClassShortcut(self):
        for i in range(10):
            val = i+48    
            if self.sender().key() == val:
                self.classList.setClass(i)
                break

    def setToolShortcut(self):
        if self.sender().key() == QKeySequence(Qt.Key_F1):
            self.selectToolButtonX(1)
        if self.sender().key() == QKeySequence(Qt.Key_F2):
            self.selectToolButtonX(2)
        if self.sender().key() == QKeySequence(Qt.Key_F3):
            self.selectToolButtonX(3)
        if self.sender().key() == QKeySequence(Qt.Key_F4):
            self.selectToolButtonX(4)
        if self.sender().key() == QKeySequence(Qt.Key_F5):
            self.selectToolButtonX(5)
        if self.sender().key() == QKeySequence(Qt.Key_F6):
            self.selectToolButtonX(6)

    def showLog(self):
        self.log_form.show()
        self.log_form.showLog(LOG_FILENAME)

    def showSettingsWindow(self):
        self.settings_form.show()
        
    def showPostProcessingWindow(self):   
        if self.LearningMode() == dlMode.Segmentation:
            self.segpostprocessing_form.show()
        elif self.LearningMode() == dlMode.Object_Counting:
            self.odpostprocessing_form.show() 
    
    def closeEvent(self, event):       
        self.settings.saveSettings(SETTINGS_FILENAME)
        QApplication.closeAllWindows()
        self.dl.interrupt()
        super(QMainWindow, self).closeEvent(event)
        
    def SaveSettings(self):
        # filename = QFileDialog.getSaveFileName(self, "Save Settings", '', "Settings File (*.json)")[0] #<- native version
        filename = saveFile("Save Settings","Settings File (*.json)", 'json')
        if filename:
            self.settings.saveSettings(filename)
        
    def LoadSettings(self):
        # filename = QFileDialog.getOpenFileName(self, "Select Settings File", '',"Settings (*.json)")[0] #<- native version
        filename = loadFile("Select Settings File", "Settings (*.json)")
        if filename:
            self.settings.loadSettings(filename)
            
    def settingsLoaded(self):
        self.ClassListUpdated()

    def closeModeWindows(self):
        if self.initialized:
            self.odpostprocessing_form.hide()
            self.segpostprocessing_form.hide()

        
    def showResultsWindow(self):
        if self.files.testImageLabelspath is None:
            self.PopupWarning('Prediction folder not selected')
            return
        
        self.results_form.show()

    def OpenManual(self):
        self.OpenWebsite('https://mianalyzer.github.io/')
    
    def OpenGithub(self):
        self.OpenWebsite('https://github.com/MIAnalyzer/MIA')
    
    def OpenWebsite(self, url):
        webbrowser.open(url) 
        
    def PopupWarning(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()  
        
    def ConfirmPopup(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return msg.exec_() == QMessageBox.Ok
        
    def ShapesChanged(self):
        for i in range (self.NumOfClasses()):
            label = self.findChild(QLabel, 'numOfContours_class' + str(i))
            if i == 0:
                text = str(len(self.canvas.painter.backgroundshapes))
            elif self.canvas.painter.shapes:
                text = str(len(self.canvas.painter.shapes.getShapesOfClass_x(i)))
            else:
                text = ""
            label.setText(text)
        if self.TrackingModeEnabled:
            self.tracking.changeTimePoint(self.canvas.painter.shapes, self.files.currentFrame if self.tracking.stackMode else self.files.currentImage)
        self.canvas.SaveCurrentLabel()
        
    def setSmartMode(self):
        assert (self.LearningMode() == dlMode.Segmentation)
        self.canvas.painter.smartmode = self.CBSmartMode.isChecked()
        self.setCanvasFocus()
        
    @property
    def allowInnerContours(self):
        return self.CBInner.isChecked()
        
    def changeLearningMode(self, i):
        self.dl.WorkingMode = dlMode(i)
        self.canvas.setCanvasMode(self.LearningMode())
        self.closeModeWindows()
        if self.LearningMode() == dlMode.Segmentation:
            self.SegmentationSettings.show()
        else:
            self.SegmentationSettings.hide()
            
        if self.LearningMode() == dlMode.Object_Counting:
            self.DetectionSettings.show()
        else:
            self.DetectionSettings.hide()
        
        if self.LearningMode() == dlMode.Classification:
            self.ClassificationToolButtons.show()
        else:
            self.ClassificationToolButtons.hide()

        if self.initialized:
            # could be done in an observer
            self.results_form.ModeChanged()
            self.training_form.ModeChanged()
            if not self.TrackingSupported():
                self.enableTrackingMode(False)

            self.segpostprocessing_form.enableTrackingMode()
            self.odpostprocessing_form.enableTrackingMode()
            

        self.setWorkingFolder()

    def TrackingSupported(self):
        if self.LearningMode() == dlMode.Segmentation:
            return True
        if self.LearningMode() == dlMode.Object_Counting:
            return True
        else:
            return False

    def loadTrack(self):
        self.Tracking.setEnabled(False)
        if not self.files.testImagespath or not self.files.testImageLabelspath:
            return
        if self.TrackingModeEnabled and not self.files.train_test_dir:
            self.tracking.loadTrack()
            self.Tracking.setEnabled(True)
        
    def calcTracking(self):
        if not self.TrackingModeEnabled or not self.files.testImagespath or not self.files.testImageLabelspath:
            self.PopupWarning('No predicted files')
            return
        with self.wait_cursor():
            self.tracking.performTracking()
            self.canvas.ReloadImage()

    def enableTrackingMode(self, enable):
        if self.TrackingSupported():
            self.Tracking.setVisible(enable)
            self.loadTrack()
            self.canvas.ReloadLabels()
        else:
            self.Tracking.setVisible(False)
        
    def drawObjectTrack(self):
        return self.TrackingModeEnabled and self.CBshowtrack.isChecked()
            
    def colorCodeObjects(self):
        return self.TrackingModeEnabled and self.CBtrackcolor.isChecked()
    
    @property
    def TrackingModeEnabled(self):
        if not self.files.train_test_dir and self.TrackingSupported():
            return self.Tracking.isVisible()
        return False

    def setToolButtons(self):
        if self.activeClass() == 0:
            self.canvas.painter.enableDrawBackgroundMode()                
        else:
            self.canvas.painter.disableDrawBackgroundMode()
            
        # we need to first set all to invisible as otherwise the widget_width is extended 
        # and somewhat hard to control with pyqt :(
        for tool in canvasToolButton:
            widget = self.findChild(DCDButton, tool.name)
            if not widget:
                continue
            widget.setVisible(False)
        for tool in self.canvas.painter.tools: 
            widget = self.findChild(DCDButton, tool.name)
            if not widget:
                continue
            widget.setVisible(True)

        self.ToolButtons.setVisible(True)
        show = False
        count = 0
        for index in range(self.ToolButtonslayout.count()):
            if self.ToolButtonslayout.itemAt(index).widget().isVisible():
                count += 1
                if count >= 2:
                    show = True
        self.ToolButtons.setVisible(show)

        if self.canvas.tool.type not in self.canvas.painter.tools:
            self.setCanvasTool(canvasTool.drag.name)



        
    def addClass(self):
        self.classList.addClass()
        
    def removeLastClass(self):
        self.classList.removeClass()     
        
    def activeClass(self):
        c = self.classList.getActiveClass()
        assert(c >= 0)
        return c
    
    def ClassColor(self, c = None):
        if c is None:
            c = self.activeClass()
        color = self.classList.getClassColor(c)
        color.setAlpha(255)
        return color
    
    def NumOfClasses(self):
        return self.classList.getNumberOfClasses()
    
    def setNumberOfClasses(self, numclasses):
        self.classList.setNumberOfClasses(numclasses)

    def setCanvasFocus(self):
        self.canvas.setFocus()
    
    def updateImage(self):
        self.canvas.checkForChanges()
        self.setCanvasFocus()
        
    def classcolorChanged(self):
        self.canvas.redrawImage()
        self.updateCursor()

    def classChanged(self):
        self.setToolButtons()
        self.updateCursor()
        self.canvas.painter.resetSmartMask()

    def updateCursor(self):
        self.canvas.updateCursor()
        self.setCanvasFocus()

    def clear(self, resetview = True):
        self.canvas.clearLabel()
        self.deleteLabel()
        self.canvas.ReloadLabels(resetview)
        QApplication.processEvents()

    def undo(self):
        self.canvas.undoDrawing()

    def checkIfUndoPossible(self):
        if self.canvas.painter.undoPossible():
            self.Bundo.setEnabled(True)
        else:
            self.Bundo.setEnabled(False)
    
    def deleteLabel(self):
        if self.files.CurrentLabelPath() and os.path.exists(self.files.CurrentLabelPath()):
            os.remove(self.files.CurrentLabelPath())
        
    def nextImage(self):
        self.files.nextImage()
        self.changeImage()

    def previousImage(self):
        self.files.previousImage()
        self.changeImage()
            
    def changeImage(self, forcereset=False):
        if self.files.numofImages > 0:
            self.readCurrentImageAsBGR(forcereset)
        else:
            # width = self.canvas.geometry().width()
            # height = self.canvas.geometry().height()
            self.canvas.reset()
            self.currentImageFile = None
        self.setImageText() 

    def setImageText(self):
        if self.files.currentImage == -1:
            text = 'No Image Displayed'
        else:
            if not self.files.CurrentImageName() or not self.currentImageFile:
                return
            text = "%d of %i: " % (self.files.currentImage+1,self.files.numofImages) + self.files.CurrentImageName(True) 
            if self.currentImageFile.isStack():
                text = text + " " + "%d/%i: " % (self.files.currentFrame+1,self.currentImageFile.numOfImagesInStack())
        self.StatusFile.setText(text)
          
    def setTrainFolder(self):
        # folder = str(QFileDialog.getExistingDirectory(self, "Select Training Image Folder"))
        folder = openFolder("Select Training Image Folder")
        if not folder:
            return
        
        self.files.setTrainFolder(folder)
        self.setWorkingFolder()
        
    def setTestFolder(self):
        # folder = str(QFileDialog.getExistingDirectory(self, "Select Test Image Folder"))
        folder = openFolder("Select Test Image Folder")
        if not folder:
            return
        self.files.setTestFolder(folder)
        self.setWorkingFolder()
       
    def switchToTrainFolder(self):
        self.files.switchToTrainFolder()
        self.setWorkingFolder()
        
    def switchToTestFolder(self):
        self.files.switchToTestFolder()
        self.setWorkingFolder()
            
    def setWorkingFolder(self):
        self.files.getFiles()
        
        if self.files.currentimagesrootpath:
            self.loadTrack()
            model = QFileSystemModel()
            model.setRootPath(self.files.currentimagesrootpath)
            self.TVFiles.setModel(model)
            self.TVFiles.setRootIndex(model.index(self.files.currentimagesrootpath))
            self.TVFiles.setVisible(True)
        else:
            self.TVFiles.setVisible(False)
            
        self.changeImage(forcereset = True)
        self.setFolderLabels()
        
    def FileExplorerItemSelected(self, index):
        file = index.model().fileName(index)
        path = index.model().filePath(index)
        if self.files.moveToSubFolder(os.path.dirname(os.path.abspath(path))):
            self.loadTrack()
        if self.files.setImagebyName(file):
            self.changeImage()


    def ensureLabelFolder(self):
        self.files.ensureLabelFolder()


    def setFolderLabels(self):
        if self.files.trainImagespath:
            self.BTrainImageFolder.setText('...' + self.files.trainImagespath[-18:])
            self.BTrainImageFolder.setEnabled(True)
        if self.files.testImagespath:
            self.BTestImageFolder.setText('...' + self.files.testImagespath[-20:])
            self.BTestImageFolder.setEnabled(True)
        if self.files.train_test_dir:
            if self.BTrainImageFolder.isEnabled():
                self.BTrainImageFolder.setStyleSheet('font:bold; text-align:left')
                self.BTestImageFolder.setStyleSheet('font:normal;text-align:left')
        else:
            if self.BTestImageFolder.isEnabled():
                self.BTrainImageFolder.setStyleSheet('font:normal;text-align:left')
                self.BTestImageFolder.setStyleSheet('font:bold;text-align:left')
    
    def ClassListUpdated(self):
        if self.initialized:
            self.training_form.SBClasses.SpinBox.setValue(self.NumOfClasses())
            self.training_form.settings_form.changeClassWeightSettings()
      
    @property
    def separateStackLabels(self):
        return self.dl.data.separateStackLabels
    
    @separateStackLabels.setter
    def separateStackLabels(self, value):
        self.dl.data.separateStackLabels = value

    def enableCopyStackLabels(self):
        self.BCopyStackLabel.setEnabled(self.separateStackLabels)

    def copyStackLabels(self):
        if self.files.CurrentLabelPath() and os.path.exists(self.files.CurrentLabelPath()) and self.hasStack():
            if self.ConfirmPopup('Are you sure? All labels of the current stack are replaced.'):
                self.files.copyStackLabels(self.files.CurrentLabelPath(), self.currentImageFile.numOfImagesInStack())

    def hasStack(self):
        if self.currentImageFile is None:
            return False
        return self.currentImageFile.isStack()
            
    def hasImage(self):
        return self.currentImageFile is not None
         
    def setCanvasMode(self):
        tool = self.sender().objectName()
        self.setCanvasTool(tool)
        
    def setCanvasTool(self, tool):
        self.canvas.setnewTool(tool)

    def selectToolButtonX(self,x):
        i = 0
        for b in canvasToolButton:
            widget = self.findChild(DCDButton, b.name)
            if not widget:
                continue
            if widget.isVisible():
                i += 1
            if i == x:
                self.setCanvasTool(widget.objectName())
                return

    def readCurrentImageAsBGR(self, forcereset=False):
        with self.wait_cursor():
            
            keep = False
            frame = 0
            if not forcereset and self.currentImageFile and self.keepImageSettings:
                br = self.currentImageFile.brightness
                c = self.currentImageFile.contrast
                frame = self.files.currentFrame
                keep = True
            
            try:
                self.currentImageFile = ImageFile(self.files.CurrentImagePath(), asBGR = True)
            except:
                self.PopupWarning('Can not load image')
                return
                
            # order important
            if self.currentImageFile.isStack():
                self.StackSettings.show()
                self.initFrameSlider(frame)           
            else:
                self.StackSettings.hide()
                
            self.resetBrightnessContrast()
            if keep:
                self.currentImageFile.brightness = br
                self.currentImageFile.contrast = c
           
            self.canvas.ReloadImage(not keep)


    def initFrameSlider(self, frame):
        maxval = self.currentImageFile.numOfImagesInStack()
        self.SFrame.setMaximum(maxval)
        if frame >= maxval:
            frame = 0
        self.SFrame.setValue(maxval - frame)
        self.canvas.resetView()     
        
    def nextFrame(self):
        self.SFrame.setValue(self.SFrame.value()-1)
        
    def previousFrame(self):
        self.SFrame.setValue(self.SFrame.value()+1)

    def FrameChanged(self):
        self.files.currentFrame = self.currentImageFile.numOfImagesInStack() - self.SFrame.value() 
        self.canvas.ReloadImage(resetView = False)
        self.setImageText()
        
    def setBrightness(self):
        if not self.currentImageFile:
            return
        self.currentImageFile.brightness = self.SBrightness.value()
        self.canvas.ReloadImage(resetView = False)
        
    def setContrast(self):
        if not self.currentImageFile:
            return
        c = self.SContrast.value()
        if c > 1:
            self.currentImageFile.contrast = 1 + c/10 
        else:
            self.currentImageFile.contrast = 1 + c/51
        self.canvas.ReloadImage(resetView = False)
        
    def keepImageSettingsChanged(self):
        self.keepImageSettings = self.CBKeep.isChecked()
        
    def resetBrightnessContrast(self):
        self.SContrast.setValue(0)
        self.SBrightness.setValue(0)
        
        self.setContrast()
            
    def getCurrentImage(self):   
        if self.currentImageFile:
            return self.currentImageFile.getCorrectedImage(self.files.currentFrame)
        else:
            return None
    
    def setWorkers(self, numWorker):
        self.maxworker = numWorker
        self.dl.worker = numWorker
    
    def emitinitProgress(self, num):
        self.initProgress.emit(num)
        
    def emitProgress(self):
        self.Progress.emit()
        
    def emitProgressFinished(self):
        self.ProgressFinished.emit()
        
    def emitBlockWindow(self):
        self.Block.emit()
        
    def emitUnBlockWindow(self):
        self.UnBlock.emit()
        
    def emitPopup(self, message):
        self.Popup.emit(message)
        
    def emitStatus(self, message):
        self.StatusMessage.emit(message)
    
    
    @pyqtSlot(int)
    def initializeProgress(self, ticks):
        assert(ticks != 0)
        self.progresstick = 1/ticks *100
        self.progress = 0
        self.setProgress(0)
     
    @pyqtSlot()  
    def addProgress(self):
        with self.lock:
            self.progress += self.progresstick
            
        if self.progresslock.locked():
            return
        with self.progresslock:
            self.setProgress(self.progress)
    
    @pyqtSlot()
    def finishProgress(self):
        self.setProgress(100)
              
    @pyqtSlot()
    def blockWindow(self):
        self.setEnabled(False)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        QApplication.processEvents()
        
    @pyqtSlot()
    def unblockWindow(self):
        self.setEnabled(True)
        QApplication.restoreOverrideCursor()
        QApplication.processEvents()

    def setProgress(self,value):
        if value < 100:
            self.StatusProgress.show()
            self.StatusProgress.setValue(value)
        else:
            self.StatusProgress.hide()
        self.processEvents()

    def processEvents(self):
        if self.processlock.locked():
            return
        with self.processlock:
            QApplication.processEvents()
    
    def writeStatus(self,msg):
        self.Status.setText(msg)

            
    def toggleTrainStatus(self, training):
        self.lockModel(training)
        self.training_form.setParameterStatus()
        self.training_form.toggleTrainStatus(training=training)
        self.plotting_form.toggleTrainStatus(training=training)

    def togglePredictionStatus(self, prediction):
        self.lockModel(prediction)
        self.training_form.toggleTrainStatus(training=prediction)

    def lockModel(self, lock):
        self.Bloadmodel.setEnabled(not lock)
        self.Bresetmodel.setEnabled(not lock)
    
    def setClassWeights(self): 
        if self.NumOfClasses() > 2:
            weights = np.zeros((self.NumOfClasses()), dtype = np.float)
            for i in range(self.NumOfClasses()):
                weights[i] = self.classList.getClassWeight(i)
        else:
            weights = self.classList.getClassWeight(1)            
        self.dl.data.setClassWeights(weights)
        

    ## deep learning  
    def LearningMode(self):
        return self.dl.WorkingMode

    def loadModel(self):
        if self.saveload_modelweights_only:
            filename = loadFile("Select Model File", "Model (*.h5)")
            if filename:
                with self.wait_cursor():
                    try:
                        self.settings.loadSettings(filename.replace("h5", "json"))
                    except:
                        pass
                    self.dl.initModel(self.NumOfClasses())

                    self.dl.LoadModelWeights(filename)
                    self.writeStatus('model loaded')
                    
        else:
            filename = openFolder('Select a Model Directory')
            if filename:
                with self.wait_cursor():
                    self.dl.LoadModel(filename)
                    self.writeStatus('model loaded')
                    try:
                        settings_file = os.path.join(filename, 'settings.json')
                        self.settings.loadSettings(settings_file)
                    except:
                        pass


    def saveModel(self):
        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
            

        if self.saveload_modelweights_only:
            filename = saveFile("Save Model To File","Model File (*.h5)", 'h5')
            if filename:
                with self.wait_cursor():
                    self.dl.SaveModelWeights(filename)
                    self.settings.saveSettings(filename.replace("h5", "json"), networksettingsonly=True)
                    self.writeStatus('model saved')
        else: 
            filename = openFolder('Select a Folder as Model Directory')
            
            if filename:
                with self.wait_cursor():
                    self.dl.SaveModel(filename)
                    settings_file = os.path.join(filename, 'settings.json')
                    self.settings.saveSettings(settings_file, networksettingsonly=True)
                    self.writeStatus('model saved')

    def resetModel(self):
        if self.ConfirmPopup('Are you sure to reset model?'):
            self.dl.Reset()
            self.training_form.setParameterStatus()
            self.writeStatus('model reset')
    
    def showTrainingWindow(self):
        if self.files.trainImagespath is None:
            self.PopupWarning('Training folder not selected')
            return
 
        self.training_form.show()
        
    def startTraining(self):
        if not self.dl.parameterFit(self.NumOfClasses()):
            if not self.dl.initModel(self.NumOfClasses()):
               self.PopupWarning('Cannot initialize model') 
               return
        self.plotting_form.show()
        self.plotting_form.initialize()
        self.toggleTrainStatus(True)
        self.dl.startTraining(self.files.trainImagespath, self.files.validationpath)

    def resumeTraining(self):
        self.toggleTrainStatus(True)
        if not self.dl.resumeTraining():
            self.toggleTrainStatus(False)
        
    def predictAllImages(self):
        if self.files.testImagespath is None:
            self.PopupWarning('Prediction folder not selected')
            return

        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
        if not self.ConfirmPopup('Are you sure to predict all images in the test folder?'):
            return

        self.emitinitProgress(len(self.files.TestImages))
        self.togglePredictionStatus(True)
        thread = WorkerThread(self.predictAllImages_async, callback_end=self.PredictionFinished)
        thread.daemon = True
        thread.start()


    def PredictionFinished(self):
        self.writeStatus(str(len(self.files.TestImages)) + ' images predicted')
        self.canvas.ReloadImage()
        self.togglePredictionStatus(False)

    def predictAllImages_async(self):
        for image in self.files.TestImages:
            self.predictSingleImageOrStack(image)
        self.emitProgressFinished()

    def predictAllImages_concurrent(self):
        self.emitinitProgress(len(self.files.TestImages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxworker) as executor:
            executor.map(self.predictSingleImageOrStack,images)

        self.emitProgressFinished()
        self.switchToTestFolder()
        self.writeStatus(str(len(images)) + ' images predicted')
        
    def predictSingleImageOrStack(self, imagepath):
        # in worker thread
        image = ImageFile(imagepath)
        image.normalizeImage()
        name = self.files.getFilenameFromPath(imagepath)
        if image.isStack():
            path = self.files.extendLabelPathByFolder(self.files.testImageLabelspath, name)
        else:
            path = self.files.testImageLabelspath
            
        if self.dl.data.stacklabels == dlStackLabels.unique_collectiveInput:
            img = image.getDLInputImage(self.dl.Channels,self.dl.data.channels, -1)
            pred = self.dl.Mode.PredictImage(img)
            if image.isStack():
                filename = self.files.extendNameByMaxFrameNumber(name,image.numOfImagesInStack())
            else:
                filename = name
            self.extractAndSavePredictedResults(pred, path, filename)
        else:
            for i in range(image.numOfImagesInStack()):
                img = image.getDLInputImage(self.dl.Channels,self.dl.data.channels, i)
                pred = self.dl.Mode.PredictImage(img)
                if image.isStack():
                    filename = self.files.extendNameByFrameNumber(name,i)
                else:
                    filename = name
                self.extractAndSavePredictedResults(pred, path, filename)
        self.emitProgress()
        
    def predictImage_async(self):
        image = self.dl.data.readImage(self.files.CurrentImagePath(), self.files.currentFrame)
        if image is None:
            return
        self.clear(resetview = False)
        image = self.canvas.getFieldOfViewImage(image)
        self.dl.PredictImage_async(image)

        
    def predictImage(self):
        if self.files.CurrentImagePath() is None:
            return
        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
        with self.wait_cursor():
            image = self.dl.data.readImage(self.files.CurrentImagePath(), self.files.currentFrame)
            image = self.canvas.getFieldOfViewImage(image)
            
            if image is None:
                self.PopupWarning('Cannot load image')
                return
            prediction = self.dl.Mode.PredictImage(image)
            if prediction is None:
                self.PopupWarning('Cannot predict image')
                return

            self.extractAndUpdatePredictedResults(prediction)
            self.writeStatus('image predicted')
            
    def extractPredictedResults(self, prediction, offset = (0,0)):
        if prediction is None:
            return
        return self.dl.Mode.extractShapesFromPrediction(prediction, not self.allowInnerContours, offset)
            
    def extractAndSavePredictedResults(self, prediction, path, name):
        if prediction is None:
            return
        self.files.ensureFolder(path)
        self.dl.Mode.saveShapes(self.extractPredictedResults(prediction), os.path.join(path, (name + ".npz")))
        
    def extractAndUpdatePredictedResults(self, prediction):
        self.canvas.painter.clearFOV()
        fov = self.canvas.getFieldOfViewRect()
        shapes = self.extractPredictedResults(prediction, offset = (int(fov.x()),int(fov.y())))
        self.updateCurrentShapes(shapes)

        
    def updateCurrentShapes(self, newshapes):
        if newshapes:
            self.canvas.painter.addShapes(newshapes)
            self.canvas.checkForChanges()

    def autoSegment(self):
        with self.wait_cursor():
            self.canvas.painter.autosegment()
    
    def autoPointDetection(self):
        with self.wait_cursor():
            self.canvas.painter.autodetection()
        
    def saveDLClass(self):
        
        defaultname = 'model_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = saveFile("Save Settings","Settings File (*.pkl)", 'pkl', defaultname)
        if filename:
            filehandler = open(filename, 'wb') 
            self.settings.saveSettings(filename.replace("pkl", "json"), networksettingsonly=True)
            
            self.dl.detachObserver(self.dlobserver)
            hed = self.dl.hed
            self.dl.hed = None
            model = self.dl.Model
            self.dl.Model = None
            
            pickle.dump(self.dl, filehandler)
            self.dl.hed = hed
            self.dl.Model = model
            self.dl.attachObserver(self.dlobserver)
        
    # deep learning observer functions
    def dlStarted(self):
        self.toggleTrainStatus(True)
        
    def dlProgress(self):
        self.plotting_form.refresh()
        
    def dlEpoch(self, epoch):
        if self.files.train_test_dir or self.predictionRate == 0:
            return    
        if (epoch % self.predictionRate) == self.predictionRate - 1:
            self.predictImage_async()
        
    def dlFinished(self):
        self.toggleTrainStatus(False)

    def dlResult(self, result):
        self.canvas.painter.clearFOV()
        self.extractAndUpdatePredictedResults(result)
        # self.clear()
        # self.extractAndSavePredictedResults(result, self.files.labelpath, self.files.CurrentLabelName())
        # self.canvas.ReloadImage()

    def dlError(self, err):
        (exctype, value, traceback) = err
        if exctype == errors.ResourceExhaustedError:
            self.PopupWarning('Out of memory!') 
        else:
            MIA_UI.excepthook(exctype, value, traceback)

              
    ############################
    @contextmanager
    def wait_cursor(self):
        try:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            QApplication.processEvents()
            yield
        finally:
            QApplication.restoreOverrideCursor()

    def excepthook(excType, excValue, tracebackobj):
        
        timeString = time.strftime("%Y-%m-%d, %H:%M:%S")
        separator = '-' * 80
        
        tb = io.StringIO()
        traceback.print_tb(tracebackobj, None, tb)
        tb.seek(0)
        info = tb.read()
        errmsg = '%s: \n%s' % (str(excType), str(excValue))
        sections = [separator, timeString, separator, errmsg, separator, info]
        msg = '\n'.join(sections)
        print(msg)
        
        try:
            f = open(LOG_FILENAME, "a")
            f.write(msg)
            f.close()
        except IOError:
            pass
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText('An error occured, a logfile was created with further information. \nPlease forward logfile. \n')
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
sys.excepthook = MIA_UI.excepthook
