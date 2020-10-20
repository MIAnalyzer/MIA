# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:32:20 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm


class Window(object):
    def setupUi(self, Form):
        width = 250
        height= 30
        Form.setWindowTitle('Segmentation Settings') 
        styleForm(Form)
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        
        vlayout = QVBoxLayout(self.centralWidget)
        vlayout.setContentsMargins(3, 3, 3, 3)
        vlayout.setSpacing(2)
        
        self.CBDitanceMap = QCheckBox("Separate contours",self.centralWidget)
        self.CBDitanceMap.setObjectName('CreateWeightedDistanceMap')
        self.CBDitanceMap.setToolTip('Check to separate adjacent contours')
        
        vlayout.addWidget(self.CBDitanceMap)
        self.centralWidget.setLayout(vlayout)


class SegmentationSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SegmentationSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBDitanceMap.stateChanged.connect(self.ShapeSeparationChanged)
        
    def show(self):
        self.CBDitanceMap.setChecked(self.parent.parent.dl.Mode.useWeightedDistanceMap)
        super(SegmentationSettingsWindow, self).show()
        
    def ShapeSeparationChanged(self):
        self.parent.parent.dl.Mode.useWeightedDistanceMap = self.CBDitanceMap.isChecked()