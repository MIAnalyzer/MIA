# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:48:32 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from ui.ui_utils import DCDButton, LabelledDoubleSpinBox, LabelledSpinBox
from dl.method.mode import dlMode

class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Post Processing') 
        styleForm(Form)

        self.centralWidget = QWidget(Form)
        
        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)

        self.CBTrackingMode = QCheckBox("Tracking Mode",self.centralWidget)
        self.CBTrackingMode.setToolTip('Check to enable tracking mode')
        self.CBTrackingMode.setObjectName('TrackingMode_seg')

        self.vlayout.addWidget(self.CBTrackingMode)
        
        
        self.CBWatershed = QCheckBox("Contour separation on prediction",self.centralWidget)
        self.CBWatershed.setObjectName('WT_Separation')
        self.CBWatershed.setToolTip('Check to perform watershed separation on predictions')
        
        self.CBWTThresh = LabelledDoubleSpinBox ('Detection threshold ', self.centralWidget)
        self.CBWTThresh.setObjectName('WT_Threshold')
        self.CBWTThresh.SpinBox.setRange(0.1,0.9)
        self.CBWTThresh.SpinBox.setSingleStep(0.1)
        self.CBWTThresh.SpinBox.setDecimals(1)

        self.SBWTDist = LabelledSpinBox ('Min Contour Distance', self.centralWidget)
        self.SBWTDist.setObjectName('WT_Distance')
        self.SBWTDist.SpinBox.setRange(1,500)
        
        self.vlayout.addWidget(self.CBWatershed)
        self.vlayout.addWidget(self.CBWTThresh)
        self.vlayout.addWidget(self.SBWTDist)
        
        
        self.hlayout = QHBoxLayout()
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
        
        self.h2layout = QHBoxLayout()
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

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())
                

        

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
        
        
        
        self.CBWatershed.setChecked(self.parent.dl.seg_wt_separation)
        self.SBWTDist.SpinBox.setValue(self.parent.dl.seg_wt_mindist)
        self.CBWTThresh.SpinBox.setValue(self.parent.dl.seg_wt_thresh)
        # self.SBWTDist.setEnabled(self.parent.dl.seg_wt_separation)
        # self.CBWTThresh.setEnabled(self.parent.dl.seg_wt_separation)
        
        
        self.CBWatershed.stateChanged.connect(self.WatershedChanged)
        self.SBWTDist.SpinBox.valueChanged.connect(self.WTDistanceChanged)
        self.CBWTThresh.SpinBox.valueChanged.connect(self.WTThreshChanged)
        
    def show(self):
        self.WatershedChanged()
        super(SegmentationPostProcessingWindow, self).show()
        
        
    def showSkeleton(self):
        self.parent.canvas.drawSkeleton = self.CBCalculateSkeleton.isChecked()

    def skeletonsmoothing(self): 
        val = self.SSkeletonSmoothing.value()
        val = val + 1 if val % 2 == 0 else val
        self.parent.canvas.skeletonsmoothingfactor = val
        
    def closeEvent(self, event):
        self.parent.canvas.ReloadImage(resetView = False)
    
    def setminContourSize(self):
        self.parent.canvas.setMinimumContourSize(self.SBminContourSize.value())

    def enableTrackingMode(self):
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.parent.enableTrackingMode(self.CBTrackingMode.isChecked())
        
    def WatershedChanged(self):
        self.parent.dl.seg_wt_separation = self.CBWatershed.isChecked()
        self.SBWTDist.setEnabled(self.parent.dl.seg_wt_separation)
        self.CBWTThresh.setEnabled(self.parent.dl.seg_wt_separation)
        
        
    def WTDistanceChanged(self):
        self.parent.dl.seg_wt_mindist = self.SBWTDist.SpinBox.value()
        
    def WTThreshChanged(self):
        self.parent.dl.seg_wt_thresh = self.CBWTThresh.SpinBox.value()
        