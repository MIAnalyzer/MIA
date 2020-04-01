# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:28:08 2019

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

import io
import traceback
import glob
import os
import cv2 
import multiprocessing
import concurrent.futures

import dl.DeepLearning as DeepLearning
from ui.UI import MainWindow
from ui.Tools import canvasTool
from ui.ui_Canvas import Canvas
from ui.ui_Results import ResultsWindow
from ui.ui_Training import TrainingWindow
from ui.ui_Settings import SettingsWindow
from ui.ui_PostProcessing import PostProcessingWindow

import utils.Contour as Contour
import time

PREDICT_WORMS = False
PRELOAD = True
PRELOAD_MODEL = 'models/Worm prediction_200221.h5'
LOG_FILENAME = 'log/logfile.log'





class DeepCellDetectorUI(QMainWindow, MainWindow):
    def __init__(self):
        super(DeepCellDetectorUI, self).__init__()

        self.setupUi(self)
        self.setCallbacks()
        self.show()
        self.imagepath = None
        self.labelpath = None
        self.trainImagespath = None
        self.trainImageLabelspath = None
        self.testImagespath = None
        self.testImageLabelspath = None
        self.train_test_dir = True
        self.LearningMode = 1
        
        self.progresstick = 10
        self.progress = 0
        self.maxworker = multiprocessing.cpu_count() // 2
        
        self.files = None
        self.currentImage = 0
        self.numofImages = 0
        self.setFocusPolicy(Qt.NoFocus)
        width = self.canvas.geometry().width()
        height = self.canvas.geometry().height()    

        self.dl = DeepLearning.DeepLearning()
        # init canvas
        self.hlayout.removeWidget(self.canvas)
        self.canvas.close()
        self.canvas = Canvas(self)
        self.canvas.setMouseTracking(True)
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        self.canvas.setFixedSize(QSize(width, height))
        self.canvas.reset(width, height)
        self.hlayout.addWidget(self.canvas)
        horizontalspacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hlayout.addItem(horizontalspacer)
        self.classList.setClass(0)
        self.changeLearningMode(self.LearningMode)
        self.CBLearningMode.setCurrentIndex (self.LearningMode)

        
        ### windows - should this be moved to UI ?
        self.results_form = ResultsWindow(self)
        self.training_form = TrainingWindow(self)
        self.settings_form = SettingsWindow(self)
        self.postprocessing_form = PostProcessingWindow(self)
        
        self.updateClassList()
        
        if PREDICT_WORMS:
            self.settings_form.CBgpu.setCurrentIndex(1)
            self.settings_form.CBgpu.setEnabled(False)
            if PRELOAD:
                try:
                    self.dl.LoadModel(PRELOAD_MODEL)
                except:
                    pass
            self.Btrain.setEnabled(False)
            self.BLoadTrainImages.setEnabled(False)
            self.Bsavemodel.setEnabled(False)
            self.Baddclass.setEnabled(False)
            self.Bdelclass.setEnabled(False)
            self.classList.itemWidget(self.classList.item(1)).edit_name.setText('worm')
            self.CBLearningMode.setEnabled(False)


    def setCallbacks(self):
        self.Bclear.clicked.connect(self.clear)
        self.BLoadTrainImages.clicked.connect(self.setTrainFolder)
        self.BLoadTestImages.clicked.connect(self.setTestFolder)
        self.BTestImageFolder.clicked.connect(self.switchToTestFolder)
        self.BTrainImageFolder.clicked.connect(self.switchToTrainFolder)
        self.Bnext.clicked.connect(self.nextImage)
        self.Bprev.clicked.connect(self.previousImage)
        
        self.Bdrag.clicked.connect(self.setCanvasMode)
        self.Bdraw.clicked.connect(self.setCanvasMode)
        self.Bassign.clicked.connect(self.setCanvasMode)
        self.Bextend.clicked.connect(self.setCanvasMode)
        self.Bdelete.clicked.connect(self.setCanvasMode)
        self.Bpoly.clicked.connect(self.setCanvasMode)
        
        self.Btrain.clicked.connect(self.showTrainingWindow)
        self.Bpredictall.clicked.connect(self.predictAllImages)
        self.Bpredict.clicked.connect(self.predictImage)
        self.Bloadmodel.clicked.connect(self.loadModel)
        self.Bsavemodel.clicked.connect(self.saveModel)
        
        self.Bresults.clicked.connect(self.showResultsWindow)
        self.BSettings.clicked.connect(self.showSettingsWindow)
        self.BPostProcessing.clicked.connect(self.showPostProcessingWindow)
        
        self.Baddclass.clicked.connect(self.addClass)
        self.Bdelclass.clicked.connect(self.removeLastClass)

        self.CBLearningMode.currentIndexChanged.connect(self.changeLearningMode)
        self.CBExtend.clicked.connect(self.updateCursor)
        self.CBErase.clicked.connect(self.updateCursor)
        
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
        self.results_form.hide()
        self.training_form.hide()
        self.postprocessing_form.hide()
        self.settings_form.hide()
        
    def showResultsWindow(self):
        if self.testImageLabelspath is None:
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
        
    def numOfContoursChanged(self):
        for i in range (self.NumOfClasses()):
            label = self.findChild(QLabel, 'numOfContours_class' + str(i))
            label.setText(str(len(self.canvas.Contours.getContoursOfClass_x(i))))
        self.canvas.createLabelFromContours()
        
    def changeLearningMode(self, i):
        self.LearningMode = i
        if i == 0: # classification
            self.SegmentButtons.hide()
            self.ClassificationButtons.show()
            self.setCanvasTool(canvasTool.drag.name)
            self.canvas.enableToggleTools = False
        elif i == 1: # segmentation
            self.SegmentButtons.show()
            self.ClassificationButtons.hide()
            self.setCanvasTool(canvasTool.drag.name)
            self.canvas.enableToggleTools = True
        else:
            self.SegmentButtons.hide()
            self.ClassificationButtons.hide()
        self.setWorkingFolder()
        
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
        self.canvas.clearContours()
        self.deleteLabel()
        self.changeImage()
        QApplication.processEvents()
    
    def deleteLabel(self):
        if self.CurrentLabelFullName() and os.path.exists(self.CurrentLabelFullName()):
            os.remove(self.CurrentLabelFullName())
        
    def nextImage(self):
        self.currentImage += 1
        if (self.currentImage >= self.numofImages):
            self.currentImage = 0
        self.changeImage()
        
    def previousImage(self):
        self.currentImage -= 1
        if (self.currentImage < 0):
            self.currentImage = self.numofImages - 1
        self.changeImage()
            
    def changeImage(self):
        if self.numofImages > 0:
            self.canvas.ReloadImage()
            text = "%d of %i: " % (self.currentImage+1,self.numofImages) + self.CurrentFileName()
            self.StatusFile.setText(text)
        else:
            width = self.canvas.geometry().width()
            height = self.canvas.geometry().height()
            self.canvas.reset(width, height)
            
    def setTrainFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Training Image Folder"))
        if not folder:
            return
        self.trainImagespath = folder
        self.train_test_dir = True
        self.setWorkingFolder()
        
    def setTestFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Test Image Folder"))
        if not folder:
            return
        self.testImagespath = folder   
        self.train_test_dir = False
        self.setWorkingFolder()
        
    def createLabelFolder(self):
        if self.trainImagespath is not None:
            self.trainImageLabelspath = self.trainImagespath + "/" + self.CBLearningMode.currentText() + "_labels"
            self.ensureFolder(self.trainImageLabelspath)
        
        if self.testImagespath is not None:
            self.testImageLabelspath = self.testImagespath + "/" + self.CBLearningMode.currentText() + "_labels"
            self.ensureFolder(self.testImageLabelspath)
       
    def switchToTrainFolder(self):
        self.train_test_dir = True
        self.setWorkingFolder()
        
    def switchToTestFolder(self):
        self.train_test_dir = False
        self.setWorkingFolder()
            
    def setWorkingFolder(self):
        self.createLabelFolder()
        if self.train_test_dir:
            self.imagepath = self.trainImagespath
            self.labelpath = self.trainImageLabelspath
        else:
            self.imagepath = self.testImagespath
            self.labelpath = self.testImageLabelspath
        self.getFiles()
        self.changeImage()
        self.setFolderLabels()
        
    def ensureFolder(self, foldername):
        try:
            os.mkdir(foldername)
        except FileExistsError:
            pass # do nothing
        
    def setFolderLabels(self):
        if self.trainImagespath:
            self.BTrainImageFolder.setText('...' + self.trainImagespath[-18:])
            self.BTrainImageFolder.setEnabled(True)
        if self.testImagespath:
            self.BTestImageFolder.setText('...' + self.testImagespath[-20:])
            self.BTestImageFolder.setEnabled(True)
        if self.train_test_dir:
            if self.BTrainImageFolder.isEnabled():
                self.BTrainImageFolder.setStyleSheet('font:bold; text-align:left')
                self.BTestImageFolder.setStyleSheet('font:normal;text-align:left')
        else:
            if self.BTestImageFolder.isEnabled():
                self.BTrainImageFolder.setStyleSheet('font:normal;text-align:left')
                self.BTestImageFolder.setStyleSheet('font:bold;text-align:left')
        
    def getFiles(self):
        if self.imagepath is None:
            return
        exts = ['*.png', '*.bmp', '*.jpg', '*.tif']
        self.files = [f for ext in exts for f in glob.glob(os.path.join(self.imagepath, ext))]
        self.files.sort()
        self.currentImage = 0
        self.numofImages = len(self.files)
    
    def updateClassList(self):
        try:
            self.training_form.SBClasses.setValue(self.NumOfClasses())
        except:
            pass
        
    def loadImage(self):
        path = QFileDialog.getOpenFileName(None, self.tr('Select file'), "", self.tr('Image Files(*.png *.jpg *.bmp *.tif)'))
        self.canvas.image = QPixmap(path[0])
        self.canvas.ReloadImage()
         
    def setCanvasMode(self):
        tool = self.sender().objectName()[1:]
        self.setCanvasTool(tool)
        
    def setCanvasTool(self, tool):
        self.canvas.setnewTool(tool)
            
    def keyPressEvent(self, e):
        if e.key() == 32: # space bar
            self.canvas.toggleDrag()
        if 48 <= e.key() <= 57:
            classnum = e.key()-48
            if classnum < self.classList.getNumberOfClasses():
                self.classList.setClass(e.key()-48)

    def CurrentFileName(self):
        if self.CurrentFilePath() is None:
            return None
        return os.path.splitext(os.path.basename(self.CurrentFilePath()))[0]
    
    def CurrentFilePath(self):
        if self.files is None:
            return None
        return self.files[self.currentImage]
        
    def CurrentLabelFullName(self):
        if self.labelpath is None or self.CurrentFileName() is None:
            return None
        return os.path.join(self.labelpath, self.CurrentFileName()) + ".npz"
    
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
    def loadModel(self):
        filename = QFileDialog.getOpenFileName(self, "Select Model File", '',"Model (*.h5)")[0]
        if filename:
            self.dl.LoadModel(filename)
        self.writeStatus('model loaded')
    def saveModel(self):
        if not self.dl.initialized():
            self.PopupWarning('No model trained/loaded')
            return
        filename = QFileDialog.getSaveFileName(self, "Save Model To File", '', "Model File (*.h5)")[0]
        if filename.strip():
            self.dl.SaveModel(filename)
    
    def showTrainingWindow(self):
        if self.trainImagespath is None or self.trainImageLabelspath is None:
            self.PopupWarning('Training folder not selected')
            return
 
        self.training_form.show()
        
    def predictAllImages(self):
        if self.testImagespath is None or self.testImageLabelspath is None:
            self.PopupWarning('Prediction folder not selected')
            return

        if not self.dl.initialized():
            self.PopupWarning('No model trained/loaded')
            return
        
        images = glob.glob(os.path.join(self.testImagespath,'*.*'))
        images = [x for x in images if (x.endswith(".tif") or x.endswith(".bmp") or x.endswith(".jpg") or x.endswith(".png"))]

        self.initProgress(len(images))
        print('start')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxworker) as executor:
            executor.map(self.predictSingleImageFromStack,images)

        self.ProgressFinished()
        self.switchToTestFolder()
        
    def predictSingleImageFromStack(self, imagepath):
        image = cv2.imread(imagepath, cv2.IMREAD_UNCHANGED)
        pred = self.dl.PredictImage(image)
        name = os.path.splitext(os.path.basename(imagepath))[0]
        cv2.imwrite(os.path.join(self.testImageLabelspath, (name + ".tif")) , pred)
        contours = Contour.extractContoursFromLabel(pred)
        Contour.saveContours(contours, os.path.join(self.testImageLabelspath, (name + ".npz")))
        self.addProgress()
        
    
    def predictImage(self):
        if self.CurrentFilePath() is None:
            return
        if not self.dl.initialized():
            self.PopupWarning('No model trained/loaded')
            return
        self.clear()
        image = cv2.imread(self.CurrentFilePath(), cv2.IMREAD_UNCHANGED)
        prediction = self.dl.PredictImage(image)
        if prediction is None:
            self.PopupWarning('Cannot load image')
            return
        cv2.imwrite(os.path.join(self.labelpath, (self.CurrentFileName() + ".tif")) , prediction)
        contours = Contour.extractContoursFromLabel(prediction)
        Contour.saveContours(contours, os.path.join(self.labelpath, (self.CurrentFileName() + ".npz")))
        self.canvas.ReloadImage()
              
    
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
