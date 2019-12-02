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
        self.activeClass = 0
        
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
        self.setActiveClassColor(p)
        p.drawLine(p2, p1)
        self.displayedimage.setPixmap(self.image)
        self.update()
        
    def addcircle(self, center , radius):
        p = self.getPainter()
        self.setActiveClassColor(p)
        p.setBrush(QBrush(self.colors[self.activeClass], Qt.SolidPattern))
        p.drawEllipse(center, radius, radius)
        p.setBrush(QBrush(self.colors[self.activeClass], Qt.NoBrush))
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
        self.image = QPixmap(self.parent.files[self.parent.currentImage])
        if self.image is not None:
            self.displayedimage.setPixmap(self.image)
            self.drawcontours()
    
    def drawcontours(self):
        if self.image is None:
            return
        currclass = self.activeClass
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
                self.activeClass = c.classlabel
                self.setActiveClassColor(paint)
                paint.drawPath(path)
             
        self.displayedimage.setPixmap(self.image)
        self.update()
        self.activeClass = currclass
        
    def getPainter(self):
        # painter objects can only exist once per QWidget
        p = QPainter(self.image)
        p.setPen(QPen(self.colors[self.activeClass], self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        return p
        
    def setActiveClassColor(self, painter):
        painter.setPen(QPen(self.colors[self.activeClass], self.pen_size, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
        
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
        self.parent.mode.setText(self.tool.Text) 

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
        self.testImagespath = None
        self.testImageLabelspath = None
        self.maxNumOfClasses = len(self.RBClasses)
        
        self.files = None
        self.currentImage = 0
        self.numofImages = 0
        self.setFocusPolicy(Qt.NoFocus)
        width = self.canvas.geometry().width()
        height = self.canvas.geometry().height()    
        
        self.dl = DeepLearning.DeepLearning()
        self.dl.initModel()
        
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
        self.createColors()
        
        
    def setCallbacks(self):
        self.Bclear.clicked.connect(self.clear)
        self.Bload.clicked.connect(self.setFolder)
        self.Bnext.clicked.connect(self.nextImage)
        self.Bprev.clicked.connect(self.previousImage)
        
        self.Bdrag.clicked.connect(self.setMode)
        self.Bdraw.clicked.connect(self.setMode)
        self.Bassign.clicked.connect(self.setMode)
        self.Bextend.clicked.connect(self.setMode)
        self.Bdelete.clicked.connect(self.setMode)
        self.Bpoly.clicked.connect(self.setMode)
        
        
        for i in range (len(self.RBClasses)):
            self.RBClasses[i].toggled.connect(self.changeClass)
        self.RBClasses[0].setChecked(True)
        self.Btrain.clicked.connect(self.trainModel)
        self.Bpredict.clicked.connect(self.predictContours)
        
        
    def createColors(self):
        for i in range(self.maxNumOfClasses):
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            color = self.ColorClasses[i]
            color.setStyleSheet("background-color: rgb(%s, %s, %s)" % (r,g,b))
            self.canvas.colors.append(QColor(r,g,b))
        
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
        else:
            width = self.canvas.geometry().width()
            height = self.canvas.geometry().height()
            self.canvas.reset(width, height)
            
    def setFolder(self):
        self.imagepath = str(QFileDialog.getExistingDirectory(self, "Select Folder"))
        self.getFiles()
        
        try:
            self.labelpath = self.imagepath + "/labels"
            self.testImagespath = self.imagepath + "/testImages"
            self.testImageLabelspath = self.testImagespath + "/labels"
            os.mkdir(self.testImagespath)
            os.mkdir(self.testImageLabelspath)
            os.mkdir(self.labelpath)
            
        except FileExistsError:
            pass # do nothing
        self.changeImage()
        
    def getFiles(self):
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
            
    def keyPressEvent(self, e):
        if e.key() == 32: # space bar
            self.canvas.toggleDrag()
        if 48 <= e.key() <= 57:
            self.RBClasses[e.key()-48].setChecked(True)

    def changeClass(self):
        button = self.sender()
        self.canvas.activeClass = button.classNum    
    
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
        self.dl.Train()
        
        self.predictContours()
        
    def predictContours(self):
        self.dl.TestImagesPath = self.testImagespath
        self.dl.TestLabelsPath = self.testImageLabelspath
        self.dl.Predict()
        self.imagepath = self.testImagespath
        self.labelpath = self.testImageLabelspath
        self.getFiles()
        self.changeImage()

if __name__ == "__main__":
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    dcdui = DeepCellDetectorUI()
#    sys.exit(app.exec_())
    app.exec_()