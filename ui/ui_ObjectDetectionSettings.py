# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:32:23 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm


class Window(object):
    def setupUi(self, Form):
        width = 250
        height= 60
        Form.setWindowTitle('Object Counting Settings') 
        styleForm(Form)
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
         
        vlayout = QVBoxLayout(self.centralWidget)
        vlayout.setContentsMargins(3, 3, 3, 3)
        vlayout.setSpacing(2)
        

        hlayout = QHBoxLayout(self.centralWidget)
        self.SObjectSize =  QSlider(Qt.Horizontal, self.centralWidget)
        self.SObjectSize.setMinimum(1)
        self.SObjectSize.setMaximum(10)
        self.SObjectSize.setToolTip('Use to adapt object size and distribution. Use smaller values for small and densely distributed objects')
        self.LObjSize = QLabel(self.centralWidget)
        self.LObjSize.setText('Object Size')
        hlayout.addWidget(self.SObjectSize)
        hlayout.addWidget(self.LObjSize)
        
        h2layout = QHBoxLayout(self.centralWidget)
        self.SObjectIntensity =  QSlider(Qt.Horizontal, self.centralWidget)
        self.SObjectIntensity.setMinimum(1)
        self.SObjectIntensity.setMaximum(500)
        self.SObjectIntensity.setToolTip('Use to set object peak intensity')
        self.LObjIntensity = QLabel(self.centralWidget)
        self.LObjIntensity.setText('Object Intensity')
        
        h2layout.addWidget(self.SObjectIntensity)
        h2layout.addWidget(self.LObjIntensity)

        vlayout.addLayout(hlayout)
        vlayout.addLayout(h2layout)
        
        self.centralWidget.setLayout(vlayout)

class ObjectDetectionSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ObjectDetectionSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.SObjectSize.valueChanged.connect(self.ObjectSizeChanged)
        self.SObjectIntensity.valueChanged.connect(self.ObjectIntensityChanged)
        
    def show(self):
        self.SObjectSize.setValue(self.parent.parent.dl.Mode.kernel_stdev)
        self.SObjectIntensity.setValue(self.parent.parent.dl.Mode.peak_val)
        super(ObjectDetectionSettingsWindow, self).show()
        
    def ObjectSizeChanged(self):
        self.parent.parent.dl.Mode.kernel_stdev = self.SObjectSize.value()
        
    def ObjectIntensityChanged(self):
        self.parent.parent.dl.Mode.peak_val = self.SObjectIntensity.value()
        
        
        
        