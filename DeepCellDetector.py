# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:28:08 2019

@author: Koerber
"""


import sys
import glob
import os


import numpy as np
import random

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from UI import MainWindow
from Contour import *
import DeepLearning
from Tools import canvasTool
import Tools


class Canvas(QGraphicsView):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.colors = []
        self.pen_size = 3
        self.bgcolor = QColor(0,0,0)
        self.image = None
        self.bufferimage = None
        self.zoomstep = 0
        self.tool = Tools.DragTool(self)
        self.lasttool = Tools.DrawTool(self)
        self.Contours = Contours()
        self.activeContour = None
        
        self.setCursor(Qt.OpenHandCursor)
        self.displayedimage = QGraphicsPixmapItem()
        self.scene = QGraphicsScene(self)
        self.scene.addItem(self.displayedimage)
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

    def mouseMoveEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mouseMoveEvent(e)
        super(Canvas, self).mouseMoveEvent(e)


    def mouseReleaseEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mouseReleaseEvent(e)
        super(Canvas, self).mouseReleaseEvent(e)

    def mousePressEvent(self, e):
        if not self.hasImage():
            return
        self.tool.mousePressEvent(e)
        super(Canvas, self).mousePressEvent(e)                   

    def wheelEvent(self,e):
        if self.hasImage():
            self.tool.wheelEvent(e)
      
    def addline(self, p1 ,p2):
        p = self.getPainter()
        p.drawLine(p2, p1)
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def addcircle(self, center , radius):
        p = self.getPainter()
        p.setBrush(QBrush(self.parent.ClassColor(self.parent.activeClass()), Qt.SolidPattern))
        p.drawEllipse(center, radius, radius)
        p.setBrush(QBrush(self.parent.ClassColor(self.parent.activeClass()), Qt.NoBrush))
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def addPoint2Contour(self, p_image):
        if (p_image != self.activeContour.getLastPoint()):
            self.activeContour.addPoint(p_image)
            return True
        else:
            return False
            
    def finishContour(self):
        if self.activeContour is None:
            return
        self.activeContour.closeContour()
        if (self.activeContour.isValid()):
            self.Contours.addContour(self.activeContour)
        else:
            self.redrawImage()
        self.activeContour = None
        
    def fitInView(self, scale=True):
        rect = QRectF(self.displayedimage.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
                self.scale(factor, factor)
                self.zoomstep = 0

    def reset(self, width, height):
        self.displayedimage.setPixmap(QPixmap(width, height))
        self.displayedimage.pixmap().fill(self.bgcolor)
        
    def newImage(self):
        self.bufferimage = QPixmap(self.parent.files[self.parent.currentImage])
        self.clearContours()
        self.getContours()
        self.redrawImage()
        self.zoomstep = 0
        self.fitInView()
        
    def hasImage(self):
        return self.image is not None
        
    def zoom(self):
        width = self.geometry().width()
        height = self.geometry().height()
        factor = 1 + self.zoomstep/10 
        pix = self.image.scaled(width * factor, height * factor, Qt.KeepAspectRatio)
        self.displayedimage.setPixmap(pix)
        
    def createLabelFromContours(self):
        if not self.Contours.empty():
            self.Contours.saveContours(self.parent.CurrentLabelFullName())
        self.clearContours()
              
    def getContours(self):
        self.clearContours()
        if os.path.exists(self.parent.CurrentLabelFullName()):
            self.Contours.loadContours(self.parent.CurrentLabelFullName())
            self.drawcontours()
    
    def redrawImage(self):
        if self.bufferimage is not None:
            self.image = self.bufferimage.copy()
            self.displayedimage.setPixmap(self.image)
            self.drawcontours()
    
    def drawcontours(self):
        if self.image is None:
            return
        paint = self.getPainter()
        for c in self.Contours.contours:  
            path = QPainterPath()
            if c.points is not None:
                for i in range(c.numPoints()):
                    if i == 0:
                        path.moveTo(np2QPoint(c.points[i].reshape(2,1)))
                    else:
                        path.lineTo(np2QPoint(c.points[i].reshape(2,1)))
                path.lineTo(np2QPoint(c.points[0].reshape(2,1)))
                self.setPainterColor(paint, self.parent.ClassColor(c.classlabel))
                paint.drawPath(path)
             
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def getPainter(self):
        # painter objects can only exist once per QWidget
        p = QPainter(self.image)
        p.setPen(QPen(self.parent.ClassColor(self.parent.activeClass()), self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        return p
        
    def setPainterColor(self, painter, color):
        painter.setPen(QPen(color, self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        
    def clearContours(self):
        self.Contours.clear()
        
    def f2intPoint(self, p):
        # we do not round here!
        # by using floor it is ensured that the mouse arrow head is inside the selected pixel
        p.setX(int(p.x()))
        p.setY(int(p.y()))
        return p
    
    def setnewTool(self, tool):
        self.lasttool = self.tool  
        
        if tool == canvasTool.drag.name:
            self.tool = Tools.DragTool(self)
        if tool == canvasTool.draw.name:
            self.tool = Tools.DrawTool(self)
        if tool == canvasTool.assign.name:
            self.tool = Tools.AssignTool(self)
        if tool == canvasTool.expand.name:
            self.tool = Tools.ExtendTool(self)
        if tool == canvasTool.delete.name:
            self.tool = Tools.DeleteTool(self)
        if tool == canvasTool.poly.name:
            self.tool = Tools.PolygonTool(self)
        
        self.setCursor(self.tool.Cursor())
        

    def toggleDrag(self):
        if self.tool.type == canvasTool.drag:
            self.setnewTool(self.lasttool.type.name)
        else:
            self.setnewTool(canvasTool.drag.name)
    

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
        self.classList.SetClass(0)
        
        
    def setCallbacks(self):
        self.Bclear.clicked.connect(self.clear)
        self.BLoadTrainImages.clicked.connect(self.setTrainFolder)
        self.BLoadTestImages.clicked.connect(self.setTestFolder)
        self.BTestImageFolder.clicked.connect(self.switchToTestFolder)
        self.BTrainImageFolder.clicked.connect(self.switchToTrainFolder)
        self.Bnext.clicked.connect(self.nextImage)
        self.Bprev.clicked.connect(self.previousImage)
        
        self.Bdrag.clicked.connect(self.setMode)
        self.Bdraw.clicked.connect(self.setMode)
        self.Bassign.clicked.connect(self.setMode)
        self.Bextend.clicked.connect(self.setMode)
        self.Bdelete.clicked.connect(self.setMode)
        self.Bpoly.clicked.connect(self.setMode)
        
        self.Btrain.clicked.connect(self.trainModel)
        self.Bpredict.clicked.connect(self.predictContours)
        self.Baddclass.clicked.connect(self.addClass)
        self.Bdelclass.clicked.connect(self.removeLastClass)
        
    def addClass(self):
        self.classList.addClass()
        
    def removeLastClass(self):
        self.classList.removeClass()
        
    def activeClass(self):
        c = self.classList.getActiveClass()
        assert(c >= 0)
        return c
    
    def ClassColor(self, c):
        return self.classList.getClassColor(c)
    
    def NumOfClasses(self):
        return self.classList.getNumberOfClasses
        
    def colorChanged(self):
        self.canvas.redrawImage()
        
    def clear(self):
        self.canvas.clearContours()
        if self.CurrentLabelFullName() and os.path.exists(self.CurrentLabelFullName()):
            os.remove(self.CurrentLabelFullName())
        self.changeImage()
        
    def nextImage(self):
        self.canvas.createLabelFromContours()
        self.currentImage += 1
        if (self.currentImage >= self.numofImages):
            self.currentImage = 0
        self.changeImage()
        
    def previousImage(self):
        self.canvas.createLabelFromContours()
        self.currentImage -= 1
        if (self.currentImage < 0):
            self.currentImage = self.numofImages - 1
        self.changeImage()
            
    def changeImage(self):
        if self.numofImages > 0:
            self.canvas.newImage()
            self.StatusFile.setText(self.CurrentFileName())
        else:
            width = self.canvas.geometry().width()
            height = self.canvas.geometry().height()
            self.canvas.reset(width, height)
            
    def setTrainFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Training Image Folder"))
        if not folder:
            return
        self.trainImagespath = folder
        self.trainImageLabelspath = self.trainImagespath + "/labels"
        self.ensureFolder(self.trainImageLabelspath)
        self.train_test_dir = True
        self.setWorkingFolder()
        
    def setTestFolder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Test Image Folder"))
        if not folder:
            return
        self.testImagespath = folder
        self.testImageLabelspath = self.testImagespath + "/labels"
        self.ensureFolder(self.testImageLabelspath)
        self.train_test_dir = False
        self.setWorkingFolder()
       
    def switchToTrainFolder(self):
        self.train_test_dir = True
        self.setWorkingFolder()
        
    def switchToTestFolder(self):
        self.train_test_dir = False
        self.setWorkingFolder()
            
    def setWorkingFolder(self):
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
        exts = ['*.png', '*.jpg', '*.tif']
        self.files = [f for ext in exts for f in glob.glob(os.path.join(self.imagepath, ext))]
        self.currentImage = 0
        self.numofImages = len(self.files)
        
    def loadImage(self):
        path = QFileDialog.getOpenFileName(None, self.tr('Select file'), "", self.tr('Image Files(*.png *.jpg *.bmp *.tif)'))
        self.canvas.image = QPixmap(path[0])
        self.canvas.newImage()
         
    def setMode(self, mode):
        tool = self.sender().objectName()[1:]
        self.canvas.setnewTool(tool)
        self.StatusMode.setText('Current Mode: ' + self.canvas.tool.Text)      
            
    def keyPressEvent(self, e):
        if e.key() == 32: # space bar
            self.canvas.toggleDrag()
        if 48 <= e.key() <= 57:
            classnum = e.key()-48
            if classnum < self.classList.getNumberOfClasses():
                self.classList.SetClass(e.key()-48)
    
    def CurrentFileName(self):
        if self.files is None:
            return None
        return os.path.splitext(os.path.basename(self.files[self.currentImage]))[0]
        
    def CurrentLabelFullName(self):
        if self.labelpath is None or self.CurrentFileName() is None:
            return None
        return os.path.join(self.labelpath, self.CurrentFileName()) + ".npz"
    
    ## deep learning  
    def trainModel(self):
        if self.imagepath is None or self.labelpath is None:
            return
        self.canvas.createLabelFromContours()
        self.dl.TrainPath = self.imagepath
        self.dl.LabelPath = self.labelpath
        if not self.dl.initialized:
            self.dl.initModel(self.classList.getNumberOfClasses())
        self.dl.Train()
        self.predictContours()
        
    def predictContours(self):
        self.dl.TestImagesPath = self.testImagespath
        self.dl.TestLabelsPath = self.testImageLabelspath
        self.dl.Predict()
        self.switchToTestFolder()
        

if __name__ == "__main__":
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    dcdui = DeepCellDetectorUI()
#    sys.exit(app.exec_())
    app.exec_()