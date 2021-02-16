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
from dl.method.mode import dlMode

class Window(object):
    def setupUi(self, Form):
        width = 180
        height= 100
        Form.setWindowTitle('Post Processing') 
        styleForm(Form)

        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        
        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)

        self.CBTrackingMode = QCheckBox("Tracking Mode",self.centralWidget)
        self.CBTrackingMode.setToolTip('Check to enable tracking mode')
        self.CBTrackingMode.setObjectName('TrackingMode_seg')
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.SBminContourSize = QSpinBox(self.centralWidget)
        self.SBminContourSize.lineEdit().setObjectName('')
        self.SBminContourSize.setRange(3,999999999)
        self.SBminContourSize.setObjectName('MinContourSize')
        self.SBminContourSize.setToolTip('Set minimum contour size')
        self.LminContourSize = QLabel(self.centralWidget)
        self.LminContourSize.setText('min Contour Size')
        self.hlayout.addWidget(self.SBminContourSize)
        self.hlayout.addWidget(self.LminContourSize)
        
        self.vlayout.addLayout(self.hlayout)
        
        self.h2layout = QHBoxLayout(self.centralWidget)
        self.CBCalculateSkeleton = QCheckBox("Show Skeleton",self.centralWidget)
        self.CBCalculateSkeleton.setToolTip('Calculate the skeleton of each contour')
        self.CBCalculateSkeleton.setObjectName('ShowSkeleton')
        self.SSkeletonSmoothing = QSlider(Qt.Horizontal, self.centralWidget)
        self.SSkeletonSmoothing.setMinimum(1)
        self.SSkeletonSmoothing.setMaximum(41)
        self.SSkeletonSmoothing.setObjectName('SkeletonSmoothing')
        self.SSkeletonSmoothing.setToolTip('Skeleton smoothing')
        self.h2layout.addWidget(self.CBCalculateSkeleton)
        self.h2layout.addWidget(self.SSkeletonSmoothing)

        self.vlayout.addLayout(self.h2layout)

        self.BCopyStackLabels = DCDButton(self.centralWidget, "Use as Stack Label")
        self.BCopyStackLabels.setToolTip('Press to reuse current label for all frames in the current stack')

        self.vlayout.addWidget(self.BCopyStackLabels)
        

        

class SegmentationPostProcessingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SegmentationPostProcessingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.SBminContourSize.setValue(self.parent.canvas.minContourSize)
        self.CBCalculateSkeleton.setChecked(self.parent.canvas.drawSkeleton)
        self.CBCalculateSkeleton.stateChanged.connect(self.showSkeleton)
        self.CBTrackingMode.setChecked(self.parent.TrackingModeEnabled)
        self.CBTrackingMode.stateChanged.connect(self.enableTrackingMode)
        self.SSkeletonSmoothing.valueChanged.connect(self.skeletonsmoothing)
        self.SSkeletonSmoothing.setValue(self.parent.canvas.skeletonsmoothingfactor)
        self.SBminContourSize.valueChanged.connect(self.setminContourSize)     
        self.BCopyStackLabels.clicked.connect(self.shareStackLabels)
        
    def showSkeleton(self):
        self.parent.canvas.drawSkeleton = self.CBCalculateSkeleton.isChecked()

    def skeletonsmoothing(self): 
        val = self.SSkeletonSmoothing.value()
        val = val + 1 if val % 2 == 0 else val
        self.parent.canvas.skeletonsmoothingfactor = val
        
    def closeEvent(self, event):
        self.parent.canvas.ReloadImage()
    
    def setminContourSize(self):
        self.parent.canvas.setMinimumContourSize(self.SBminContourSize.value())

    def enableTrackingMode(self):
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.parent.enableTrackingMode(self.CBTrackingMode.isChecked())

    def shareStackLabels(self):
        self.parent.copyStackLabels()