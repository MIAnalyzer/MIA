# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:02 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import tensorflow as tf

class Window(object):
    def setupUi(self, Form):
        width = 150
        height= 130
        Form.setWindowTitle('Settings') 
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")

        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        
        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        
        self.CBgpu = QComboBox(self.centralWidget)

        devices = tf.config.experimental.list_physical_devices()
        self.CBgpu.addItem("default")
        for dev in devices:
            self.CBgpu.addItem(dev.name)                  
        self.CBgpu.addItem("all gpus")
        self.vlayout.addWidget(self.CBgpu)     
        
        
        
        self.CBContourNumbers = QCheckBox("Show Contour Numbers",self.centralWidget)
        self.CBContourNumbers.setToolTip('Select to show contour number below each contour')
        self.vlayout.addWidget(self.CBContourNumbers)
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.SBFontSize = QSpinBox(self.centralWidget)
        self.SBFontSize.setToolTip('Set font size')
        self.SBFontSize.setRange(8,32)
        self.LFontSize = QLabel(self.centralWidget)
        self.LFontSize.setText('Font Size')
        self.hlayout.addWidget(self.SBFontSize)
        self.hlayout.addWidget(self.LFontSize)
        self.vlayout.addLayout(self.hlayout)
        
        
        
        self.h1layout = QHBoxLayout(self.centralWidget)
        self.SBPenSize = QSpinBox(self.centralWidget)
        self.SBPenSize.setRange(1,15)
        self.SBPenSize.setToolTip('Set pen size')
        self.LPenSize = QLabel(self.centralWidget)
        self.LPenSize.setText('Pen Size')
        self.h1layout.addWidget(self.SBPenSize)
        self.h1layout.addWidget(self.LPenSize)
        self.vlayout.addLayout(self.h1layout)
        
        self.h2layout = QHBoxLayout(self.centralWidget)
        self.STransparency = QSlider(Qt.Horizontal, self.centralWidget)
        self.STransparency.setToolTip('Set transparency of contours')
        self.STransparency.setMinimum(0)
        self.STransparency.setMaximum(255)
        self.LTransparency = QLabel(self.centralWidget)
        self.LTransparency.setText('Contour Transparency')
        
        self.h2layout.addWidget(self.STransparency)
        self.h2layout.addWidget(self.LTransparency)
        self.vlayout.addLayout(self.h2layout)
        
        
        

class SettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBContourNumbers.setChecked(self.parent.canvas.drawContourNumber)
        self.SBPenSize.setValue(self.parent.canvas.pen_size)
        self.SBFontSize.setValue(self.parent.canvas.FontSize)
        self.STransparency.setValue(self.parent.canvas.ContourTransparency)
        
        self.CBContourNumbers.clicked.connect(self.showContourNumbers)
        self.SBPenSize.valueChanged.connect(self.setPenSize)
        self.SBFontSize.valueChanged.connect(self.setFontSize)
        self.STransparency.valueChanged.connect(self.setTransparency)
        self.CBgpu.currentIndexChanged.connect(self.setGPU)
        
        self.CBgpu.setCurrentIndex(0)
        
    def setGPU(self):
        # finds cpu and gpu
        devices = tf.config.experimental.list_physical_devices()
        gpus = tf.config.experimental.list_physical_devices('GPU') 
        dev = self.CBgpu.currentIndex() - 1 
        if devices:
            try:
                if dev == -1:
                    # tensorflow decides 
                    tf.config.experimental.set_visible_devices(gpus, 'GPU')
                elif dev >= len(devices):
                    # currently unsupported
                    self.parent.PopupWarning('Use of multiple GPUs currently not supported')
                elif devices[dev].device_type == 'CPU':
                    # set cpu
                    tf.config.experimental.set_visible_devices([], 'GPU')
                else:
                    # set gpu
                    tf.config.experimental.set_visible_devices(devices[dev], 'GPU')

            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                self.parent.PopupWarning('GPU settings only take effect upon program start')
        
    def showContourNumbers(self):
        self.parent.canvas.drawContourNumber = self.CBContourNumbers.isChecked()
        
    def setPenSize(self):
        self.parent.canvas.pen_size = self.SBPenSize.value()
        
    def setFontSize(self):
        self.parent.canvas.FontSize = self.SBFontSize.value()
        
    def setTransparency(self):
        self.parent.canvas.ContourTransparency = self.STransparency.value()
        
    def closeEvent(self, event):
        self.parent.canvas.redrawImage()