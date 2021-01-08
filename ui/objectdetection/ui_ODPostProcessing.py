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
        height= 50
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
        self.CBTrackingMode.setObjectName('TrackingMode_od')

        self.vlayout.addWidget(self.CBTrackingMode)
        

class ObjectDetectionPostProcessingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ObjectDetectionPostProcessingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.CBTrackingMode.stateChanged.connect(self.enableTrackingMode)
        self.CBTrackingMode.setChecked(self.parent.TrackingModeEnabled)

    def enableTrackingMode(self):
        if self.parent.LearningMode() == dlMode.Object_Counting:
            self.parent.enableTrackingMode(self.CBTrackingMode.isChecked())


