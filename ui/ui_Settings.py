# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:02 2020

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
        
        
        self.CBContourNumbers = QCheckBox("Show Contour Numbers",self.centralWidget)
        self.vlayout.addWidget(self.CBContourNumbers)
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.SBFontSize = QSpinBox(self.centralWidget)
        self.SBFontSize.setRange(8,32)
        self.LFontSize = QLabel(self.centralWidget)
        self.LFontSize.setText('Font Size')
        self.hlayout.addWidget(self.SBFontSize)
        self.hlayout.addWidget(self.LFontSize)
        self.vlayout.addLayout(self.hlayout)
        
        
        
        self.h1layout = QHBoxLayout(self.centralWidget)
        self.SBPenSize = QSpinBox(self.centralWidget)
        self.SBPenSize.setRange(1,15)
        self.LPenSize = QLabel(self.centralWidget)
        self.LPenSize.setText('Pen Size')
        self.h1layout.addWidget(self.SBPenSize)
        self.h1layout.addWidget(self.LPenSize)
        self.vlayout.addLayout(self.h1layout)
        
        

class SettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBContourNumbers.setChecked(self.parent.canvas.drawContourNumber)
        self.SBPenSize.setValue(self.parent.canvas.pen_size)
        self.SBFontSize.setValue(self.parent.canvas.FontSize)
        
        self.CBContourNumbers.clicked.connect(self.showContourNumbers)
        self.SBPenSize.valueChanged.connect(self.setPenSize)
        self.SBFontSize.valueChanged.connect(self.setFontSize)
        
        
    def showContourNumbers(self):
        self.parent.canvas.drawContourNumber = self.CBContourNumbers.isChecked()
        
    def setPenSize(self):
        self.parent.canvas.pen_size = self.SBPenSize.value()
        
    def setFontSize(self):
        self.parent.canvas.FontSize = self.SBPenSize.value()
        
    def closeEvent(self, event):
        self.parent.canvas.redrawImage()