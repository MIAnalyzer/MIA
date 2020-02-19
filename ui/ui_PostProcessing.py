# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:48:32 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Window(object):
    def setupUi(self, Form):
        width = 150
        height= 65
        Form.setWindowTitle('Settings') 
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")

        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        
        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.SBminContourSize = QSpinBox(self.centralWidget)
        self.SBminContourSize.setRange(3,999999999)
        self.LminContourSize = QLabel(self.centralWidget)
        self.LminContourSize.setText('min Size')
        self.hlayout.addWidget(self.SBminContourSize)
        self.hlayout.addWidget(self.LminContourSize)
        self.vlayout.addLayout(self.hlayout)
        
        self.CBCalculateSkeleton = QCheckBox("Show Skeleton",self.centralWidget)
        self.vlayout.addWidget(self.CBCalculateSkeleton)
        
        

class PostProcessingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(PostProcessingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.SBminContourSize.setValue(self.parent.canvas.Contours.minSize)
        self.CBCalculateSkeleton.setChecked(self.parent.canvas.drawSkeleton)
        self.CBCalculateSkeleton.clicked.connect(self.showSkeleton)
        self.SBminContourSize.valueChanged.connect(self.setminContourSize)
        
    def showSkeleton(self):
        self.parent.canvas.drawSkeleton = self.CBCalculateSkeleton.isChecked()
        
    def closeEvent(self, event):
        self.parent.canvas.ReloadImage()
    
    def setminContourSize(self):
        self.parent.canvas.Contours.minSize = self.SBminContourSize.value()