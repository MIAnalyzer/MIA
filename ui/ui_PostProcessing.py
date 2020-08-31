# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:48:32 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from ui.ui_utils import DCDButton


class Window(object):
    def setupUi(self, Form):
        width = 150
        height= 100
        Form.setWindowTitle('Settings') 
        styleForm(Form)

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
        self.SBminContourSize.setToolTip('Set minimum contour size')
        self.LminContourSize = QLabel(self.centralWidget)
        self.LminContourSize.setText('min Size')
        self.hlayout.addWidget(self.SBminContourSize)
        self.hlayout.addWidget(self.LminContourSize)
        

        
        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addLayout(self.hlayout)
        
        self.CBCalculateSkeleton = QCheckBox("Show Skeleton",self.centralWidget)
        self.CBCalculateSkeleton.setToolTip('Calculate the skeleton of each contour')
        self.vlayout.addWidget(self.CBCalculateSkeleton)
        
        self.BSequence = DCDButton(self.centralWidget, 'Tracking')
        self.BSequence.setToolTip('Calculate object tracking for results')
        self.BSequence.setIcon(QIcon('icons/tracking.png'))
        self.vlayout.addWidget(self.BSequence)
        

class PostProcessingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(PostProcessingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.SBminContourSize.setValue(self.parent.canvas.minContourSize)
        self.CBCalculateSkeleton.setChecked(self.parent.canvas.drawSkeleton)
        self.CBCalculateSkeleton.stateChanged.connect(self.showSkeleton)
        self.SBminContourSize.valueChanged.connect(self.setminContourSize)
        self.BSequence.clicked.connect(self.tracking)
        
    def showSkeleton(self):
        self.parent.canvas.drawSkeleton = self.CBCalculateSkeleton.isChecked()
        
    def closeEvent(self, event):
        self.parent.canvas.ReloadImage()
    
    def setminContourSize(self):
        self.parent.canvas.setMinimumContourSize(self.SBminContourSize.value())
        
    def tracking(self):
        self.parent.calcTracking()

        