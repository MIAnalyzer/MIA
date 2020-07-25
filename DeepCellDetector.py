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
import concurrent.futures
import time
from contextlib import contextmanager

import dl.DeepLearning as DeepLearning
from dl.dl_mode import dlMode
from ui.UI import MainWindow
from ui.Tools import canvasTool
from ui.ui_Canvas import Canvas
from ui.ui_Results import ResultsWindow
from ui.ui_Training import TrainingWindow
from ui.ui_Settings import SettingsWindow
from ui.ui_PostProcessing import PostProcessingWindow
from ui.ui_TrainPlot import TrainPlotWindow

import utils.Contour as Contour
import utils.Point as Point
from utils.Image import ImageFile, supportedImageFormats
from utils.Observer import QtObserver
from utils.FilesAndFolders import FilesAndFolders

from ui.ui_utils import DCDButton

import numpy as np

PREDICT_WORMS = False
PREDICT_SPINES = True
OBJECT_COUNTING = False
PRELOAD = False
PRELOAD_MODEL = 'models/Worm prediction_200221.h5'
LOG_FILENAME = 'log/logfile.log'
CPU_ONLY = False


class DeepCellDetectorUI(QMainWindow, MainWindow):
    def __init__(self):
        super(DeepCellDetectorUI, self).__init__()
        
        self.setupUi(self)
        self.setCallbacks()
        self.show()

        
        self.progresstick = 10
        self.progress = 0
        self.maxworker = 1
        
        self.files = FilesAndFolders(self)

        self.separateStackLabels = True
        self.allowInnerContours = True
        self.setFocusPolicy(Qt.NoFocus)
        width = self.canvas.geometry().width()
        height = self.canvas.geometry().height()    
        
        self.predictionRate = 0
        
        self.dl = DeepLearning.DeepLearning()
        # init observer
        self.dlobserver = QtObserver()
        self.dl.attachObserver(self.dlobserver)
        self.dlobserver.signals.progress.connect(self.dlProgress)
        self.dlobserver.signals.epoch_end.connect(self.dlEpoch)
        self.dlobserver.signals.finished.connect(self.dlFinished)
        self.dlobserver.signals.result.connect(self.dlResult)
        self.dlobserver.signals.error.connect(self.dlError)
        self.dlobserver.signals.start.connect(self.dlStarted)
        
        
        
        # init canvas
        self.hlayout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas = Canvas(self)
        self.canvas.setMouseTracking(True)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFixedSize(QSize(width, height))
        self.canvas.reset(width, height)
        self.hlayout.addWidget(self.canvas)
        self.hlayout.addWidget(self.SFrame)
        horizontalspacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hlayout.addItem(horizontalspacer)
        
        self.classList.setClass(0)
        self.changeLearningMode(self.LearningMode().value)
        self.CBLearningMode.setCurrentIndex (self.LearningMode().value)
        self.setWorkers(multiprocessing.cpu_count() // 2)
        
        ### windows - should this be moved to UI ?
        self.results_form = ResultsWindow(self)
        self.training_form = TrainingWindow(self)
        self.settings_form = SettingsWindow(self)
        self.postprocessing_form = PostProcessingWindow(self)
        self.plotting_form = TrainPlotWindow(self)
        
        self.updateClassList()
        self.classList.setClass(1)

        if CPU_ONLY:
            self.settings_form.CBgpu.setCurrentIndex(1)
            self.settings_form.CBgpu.setEnabled(False)

        if PRELOAD:
            try:
                self.dl.LoadModel(PRELOAD_MODEL)
            except:
                pass
        if PREDICT_WORMS:
            self.Btrain.setEnabled(False)
            self.BLoadTrainImages.setEnabled(False)
            self.Bsavemodel.setEnabled(False)
            self.Baddclass.setEnabled(False)
            self.Bdelclass.setEnabled(False)
            self.classList.itemWidget(self.classList.item(1)).edit_name.setText('worm')
            self.CBLearningMode.setEnabled(False)
            self.ASetTrainingFolder.setEnabled(False)
            
        if OBJECT_COUNTING:
            self.Btrain.setEnabled(False)
            self.BLoadTestImages.setEnabled(False)
            self.Bsavemodel.setEnabled(False)
            self.classList.itemWidget(self.classList.item(1)).edit_name.setText('worm egg')
            self.CBLearningMode.setCurrentIndex(1)
            self.CBLearningMode.setEnabled(False)
            self.Btrain.setEnabled(False)
            self.Bpredictall.setEnabled(False)
            self.Bpredict.setEnabled(False)
            self.Bloadmodel.setEnabled(False)
            self.Bsavemodel.setEnabled(False)
            self.Bautoseg.setEnabled(False)
            self.Bresetmodel.setEnabled(False)


        if PREDICT_SPINES:
            self.postprocessing_form.SBminContourSize.setValue(10)
            self.settings_form.SBFontSize.SpinBox.setValue(10)
            self.settings_form.CBShapeNumbers.setChecked(False)
            self.settings_form.SBPenSize.SpinBox.setValue(1)
            self.settings_form.CBSeparateLabels.setChecked(True)

    def setCallbacks(self):
        self.Bclear.clicked.connect(self.clear)
        self.BLoadTrainImages.clicked.connect(self.setTrainFolder)
        self.BLoadTestImages.clicked.connect(self.setTestFolder)
        self.BTestImageFolder.clicked.connect(self.switchToTestFolder)
        self.BTrainImageFolder.clicked.connect(self.switchToTrainFolder)
        self.Bnext.clicked.connect(self.nextImage)
        self.Bprev.clicked.connect(self.previousImage)
        self.SFrame.valueChanged.connect(self.FrameChanged)
        
        self.Bdrag.clicked.connect(self.setCanvasMode)
        self.Bdraw.clicked.connect(self.setCanvasMode)
        self.Bassign.clicked.connect(self.setCanvasMode)
        self.Bextend.clicked.connect(self.setCanvasMode)
        self.Bdelete.clicked.connect(self.setCanvasMode)
        self.Bpoly.clicked.connect(self.setCanvasMode)
        
        self.BassignClass.clicked.connect(self.assignImageClass)
        self.BsetClass.clicked.connect(self.setImageClass)
        
        self.BsetObject.clicked.connect(self.setCanvasMode)
        self.BshiftObject.clicked.connect(self.setCanvasMode)
        
        
        self.Btrain.clicked.connect(self.showTrainingWindow)
        self.Bpredictall.clicked.connect(self.predictAllImages)
        self.Bpredict.clicked.connect(self.predictImage)
        self.Bloadmodel.clicked.connect(self.loadModel)
        self.Bsavemodel.clicked.connect(self.saveModel)
        self.Bautoseg.clicked.connect(self.autoSegment)
        self.Bresetmodel.clicked.connect(self.resetModel)
        
        self.Bresults.clicked.connect(self.showResultsWindow)
        self.BSettings.clicked.connect(self.showSettingsWindow)
        self.BPostProcessing.clicked.connect(self.showPostProcessingWindow)
        
        self.Baddclass.clicked.connect(self.addClass)
        self.Bdelclass.clicked.connect(self.removeLastClass)

        self.CBLearningMode.currentIndexChanged.connect(self.changeLearningMode)
        self.CBExtend.clicked.connect(self.updateCursor)
        self.CBErase.clicked.connect(self.updateCursor)
        
        self.SBrightness.sliderReleased.connect(self.setBrightness)
        self.SContrast.sliderReleased.connect(self.setContrast)
        
        ## menu actions
        self.ASetTrainingFolder.triggered.connect(self.setTrainFolder)
        self.ASetTestFolder.triggered.connect(self.setTestFolder)
        self.AExit.triggered.connect(self.close)
        self.ASettings.triggered.connect(self.showSettingsWindow)
        
    def showSettingsWindow(self):
        self.settings_form.show()
        
    def showPostProcessingWindow(self):   
        self.postprocessing_form.show()
    
    def closeEvent(self, event):       
        QApplication.closeAllWindows()
        self.dl.interrupt()
        super(QMainWindow, self).closeEvent(event)

        
    def showResultsWindow(self):
        if self.files.testImageLabelspath is None:
            self.PopupWarning('Prediction folder not selected')
            return
        self.results_form.show()
        
    def PopupWarning(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()  
        
    def numOfShapesChanged(self):
        for i in range (self.NumOfClasses()):
            label = self.findChild(QLabel, 'numOfContours_class' + str(i))
            if self.canvas.painter.shapes:
                label.setText(str(len(self.canvas.painter.shapes.getShapesOfClass_x(i))))
            else:
                label.setText("")
        self.canvas.SaveCurrentLabel()
        
    def changeLearningMode(self, i):
        self.dl.WorkingMode = dlMode(i)
        self.setCanvasTool(canvasTool.drag.name)

        # classification is not handeld via tools as now interaction besides dragging with the image is required
        if self.LearningMode() == dlMode.Classification: 
            self.ClassificationButtons.show()
            self.canvas.enableToggleTools = False
            self.ToolButtons.setVisible(False)
        else:
            self.ClassificationButtons.hide()
            self.canvas.enableToggleTools = True
            self.ToolButtons.setVisible(True)


        self.canvas.setCanvasMode(self.LearningMode())
        self.setWorkingFolder()
        
    def setToolButtons(self):
        # we need to first set all to invisible as otherwise the widget_width is extended 
        # and somewhat hard to control with pyqt :(
        for tool in canvasTool:
            widget = self.findChild(DCDButton, tool.name)
            if not widget:
                continue
            widget.setVisible(False)
        for tool in self.canvas.painter.tools:
            widget = self.findChild(DCDButton, tool.name)
            if not widget:
                continue
            widget.setVisible(True)

    def assignImageClass(self):
        self.canvas.assignImageClass()

    def setImageClass(self):
        self.canvas.assignImageClass()
        self.nextImage()
        
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
        
    def classcolorChanged(self):
        self.canvas.redrawImage()
        self.updateCursor()

    def classChanged(self):
        self.updateCursor()

    def updateCursor(self):
        self.canvas.updateCursor()

    def clear(self):
        self.canvas.clearLabel()
        self.deleteLabel()
        self.canvas.ReloadImage()
        QApplication.processEvents()
    
    def deleteLabel(self):
        if self.files.CurrentLabelPath() and os.path.exists(self.files.CurrentLabelPath()):
            os.remove(self.files.CurrentLabelPath())
        
    def nextImage(self):
        self.files.nextImage()
        self.changeImage()
        
    def previousImage(self):
        self.files.previousImage()
        self.changeImage()
            
    def changeImage(self):
        if self.files.numofImages > 0:
            self.readCurrentImageAsBGR()
            self.setImageText()            
        else:
            width = self.canvas.geometry().width()
            height = self.canvas.geometry().height()
            self.canvas.reset(width, height)

    def setImageText(self):
        text = "%d of %i: " % (self.files.currentImage+1,self.files.numofImages) + self.files.CurrentImageName() 
        if self.currentImageFile.isStack():
            text = text + " " + "%d/%i: " % (self.files.currentFrame+1,self.currentImageFile.numOfImagesInStack())
        self.StatusFile.setText(text)
            
    def setTrainFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Training Image Folder"))
        if not folder:
            return
        self.files.setTrainFolder(folder)
        self.setWorkingFolder()
        
    def setTestFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Test Image Folder"))
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
        self.changeImage()
        self.setFolderLabels()
        
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
    
    def updateClassList(self):
        try:
            self.training_form.SBClasses.SpinBox.setValue(self.NumOfClasses())
        except:
            pass
        
    def hasStack(self):
        return self.currentImageFile.isStack()
            
    def hasImage(self):
        return self.currentImageFile is not None
         
    def setCanvasMode(self):
        tool = self.sender().objectName()
        self.setCanvasTool(tool)
        
    def setCanvasTool(self, tool):
        self.canvas.setnewTool(tool)
            
    def keyPressEvent(self, e):
        if e.key() == 32: # space bar
            self.canvas.toggleDrag()
        if 48 <= e.key() <= 57:
            classnum = e.key()-48
            self.classList.setClass(e.key()-48)

    
    def readCurrentImageAsBGR(self):   
        with self.wait_cursor():
            self.currentImageFile = ImageFile(self.files.CurrentImagePath(), asBGR = True)
            self.resetBrightnessContrast()
            if self.currentImageFile.isStack():
                self.SFrame.show()
                self.initFrameSlider()
            else:
                self.SFrame.hide()
            self.canvas.ReloadImage()

    def initFrameSlider(self):
        maxval = self.currentImageFile.numOfImagesInStack()
        self.SFrame.setMaximum(maxval)
        self.files.currentFrame = 0
        self.SFrame.setValue(maxval)
        self.canvas.resetView()     
  
    def FrameChanged(self):
        self.files.currentFrame = self.currentImageFile.numOfImagesInStack() - self.SFrame.value() 
        self.canvas.ReloadImage(resetView = False)
        self.setImageText()
        
    def setBrightness(self):
        b = self.SBrightness.value()
        self.currentImageFile.brightness = b
        self.canvas.ReloadImage(resetView = False)
        
    def setContrast(self):
        c = self.SContrast.value()
        if c > 1:
            self.currentImageFile.contrast = 1 + c/10 
        else:
            self.currentImageFile.contrast = 1 + c/10
        self.canvas.ReloadImage(resetView = False)
        
    def resetBrightnessContrast(self):
        self.SContrast.setValue(1)
        self.SBrightness.setValue(0)
            
    def getCurrentImage(self):   
        return self.currentImageFile.getCorrectedImage(self.files.currentFrame)
    
    def setWorkers(self, numWorker):
        self.maxworker = numWorker
        self.dl.worker = numWorker
    
    def initProgress(self, ticks):
        assert(ticks != 0)
        self.progresstick = 1/ticks *100
        self.progress = 0
        self.setProgress(0)
        
    def addProgress(self):
        self.progress += self.progresstick
        self.setProgress(self.progress)
    
    def ProgressFinished(self):
        self.setProgress(100)
    
    def setProgress(self,value):
        if value < 100:
            self.StatusProgress.show()
            self.StatusProgress.setValue(value)
        else:
            self.StatusProgress.hide()
        QApplication.processEvents()
            
    def writeStatus(self,msg):
        self.Status.setText(msg)

    ## deep learning  
    def LearningMode(self):
        return self.dl.WorkingMode

    def loadModel(self):
        filename = QFileDialog.getOpenFileName(self, "Select Model File", '',"Model (*.h5)")[0]
        if filename:
            with self.wait_cursor():
                self.dl.LoadModel(filename)
                self.writeStatus('model loaded')

    def saveModel(self):
        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
        filename = QFileDialog.getSaveFileName(self, "Save Model To File", '', "Model File (*.h5)")[0]
        if filename.strip():
            with self.wait_cursor():
                self.dl.SaveModel(filename)
                self.writeStatus('model saved')

    def resetModel(self):
        self.dl.Reset()
        self.training_form.toggleParameterStatus()
        self.writeStatus('model reset')
    
    def showTrainingWindow(self):
        if self.files.trainImagespath is None:
            self.PopupWarning('Training folder not selected')
            return
 
        self.training_form.show()
        
    def startTraining(self):
        if not self.dl.parameterFit(self.NumOfClasses(), self.training_form.CBMono.isChecked()):
            if not self.dl.initModel(self.NumOfClasses(), self.training_form.CBMono.isChecked()):
               self.PopupWarning('Cannot initialize model') 
               return
        self.plotting_form.show()
        self.plotting_form.initialize()
        self.training_form.toggleTrainStatus(True)
        self.dl.Train(self.files.trainImagespath,self.files.trainImageLabelspath)
        
    def predictAllImages(self):
        if self.files.testImagespath is None:
            self.PopupWarning('Prediction folder not selected')
            return

        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
        with self.wait_cursor():
            images = glob.glob(os.path.join(self.files.testImagespath,'*.*'))
            images = [x for x in images if (x.endswith(tuple(supportedImageFormats())))]

            self.initProgress(len(images))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxworker) as executor:
                executor.map(self.predictSingleImageOrStack_async,images)

            self.ProgressFinished()
            self.switchToTestFolder()
            self.writeStatus(str(len(images)) + ' images predicted')
        
    def predictSingleImageOrStack_async(self, imagepath):
        # in worker thread
        image = ImageFile(imagepath)
        image.normalizeImage()
        name = self.files.getFilenameFromPath(imagepath)
        if image.isStack():
            path = self.files.extendLabelPathByFolder(self.files.testImageLabelspath, name)
        else:
            path = self.files.testImageLabelspath
        for i in range(image.numOfImagesInStack()):
            img = image.getDLInputImage(self.dl.MonoChrome, i)
            pred = self.dl.Mode.PredictImage(img)
            if image.isStack():
                filename = self.files.extendNameByFrameNumber(name,i)
            else:
                filename = name
            self.extractAndSavePredictedResults(pred, path, filename)
        self.addProgress()
        
    def predictImage_async(self):
        image = self.dl.data.readImage(self.files.CurrentImagePath(), self.files.currentFrame)
        if image is None:
            return
        self.dl.PredictImage_async(image)

        
    def predictImage(self):
        if self.files.CurrentImagePath() is None:
            return
        if not self.dl.initialized:
            self.PopupWarning('No model trained/loaded')
            return
        with self.wait_cursor():
            self.clear()
            image = self.dl.data.readImage(self.files.CurrentImagePath(), self.files.currentFrame)
            if image is None:
                self.PopupWarning('Cannot load image')
                return
            prediction = self.dl.Mode.PredictImage(image)
            if prediction is None:
                self.PopupWarning('Cannot predict image')
                return
            self.extractAndSavePredictedResults(prediction, self.files.labelpath, self.files.CurrentLabelName())
            
            self.canvas.ReloadImage()
            self.writeStatus('image predicted')
            
    def extractAndSavePredictedResults(self, prediction, path, name):
        if prediction is None:
            return
        
        self.files.ensureFolder(path)
        contours = self.dl.Mode.extractShapesFromPrediction(prediction, not self.allowInnerContours)
        self.dl.Mode.saveShapes(contours, os.path.join(path, (name + ".npz")))

    def autoSegment(self):
        with self.wait_cursor():
            
            image = self.canvas.getFieldOfViewImage()
            fov = self.canvas.getFieldOfViewRect()
            # not using self.dl.dataloader.readImage() here as a 8-bit rgb image is required and no extra preprocessing
            # image = self.getCurrentImage()

            if image is None:
                self.PopupWarning('No image loaded')
            prediction = self.dl.AutoSegment(image)
            if self.LearningMode() == dlMode.Segmentation:
                shapes = Contour.extractContoursFromLabel(prediction, not self.allowInnerContours, offset = (int(fov.x()),int(fov.y())))
            elif self.LearningMode() == dlMode.Object_Counting:
                shapes = Point.extractPointsFromContours(prediction, self.canvas.minContourSize ,offset = (int(fov.x()),int(fov.y())))
            else:
                return
            self.canvas.painter.shapes.addShapes(shapes)
            # autosegmentation puts all contours in class 1
            self.classList.setClass(1)
            self.canvas.checkForChanges()
        
    # deep learning observer functions
    def dlStarted(self):
        self.training_form.toggleTrainStatus(training=True)
        self.plotting_form.toggleTrainStatus(training=True)
        
    def dlProgress(self):
        self.plotting_form.refresh()
        
    def dlEpoch(self, epoch):
        if self.files.train_test_dir or self.predictionRate == 0:
            return    
        if (epoch % self.predictionRate) == self.predictionRate - 1:
            self.predictImage_async()
        
    def dlFinished(self):
        self.training_form.toggleTrainStatus(training=False)
        self.training_form.toggleParameterStatus()
        self.plotting_form.toggleTrainStatus(training=False)

    def dlResult(self, result):
        self.clear()
        self.extractAndSavePredictedResults(result, self.files.labelpath, self.files.CurrentLabelName())
        self.canvas.ReloadImage()

    def dlError(self, err):
        (exctype, value, traceback) = err
        if exctype == errors.ResourceExhaustedError:
            self.PopupWarning('Out of memory!') 
        else:
            DeepCellDetectorUI.excepthook(exctype, value, traceback)

              
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
            f = open(LOG_FILENAME, "w")
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
    
sys.excepthook = DeepCellDetectorUI.excepthook
