# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:32:20 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from ui.ui_utils import LabelledSpinBox

class Window(object):
    def setupUi(self, Form):
        width = 250
        height= 125
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

        self.CBIgnoreBackgroundTiles = QCheckBox("Prefer labelled parts",self.centralWidget)
        self.CBIgnoreBackgroundTiles.setToolTip('When checked image parts which contain only background are ignored during training')
        self.CBIgnoreBackgroundTiles.setObjectName('IgnoreBackground')

        self.SBDiscard = LabelledSpinBox ('Discard up to x background tiles', self.centralWidget)
        self.SBDiscard.setObjectName('DiscardBGTiles')
        self.SBDiscard.SpinBox.setRange(1,20)

        self.SBMinPixels = LabelledSpinBox ('Minimum required labelled pixels', self.centralWidget)
        self.SBMinPixels.setObjectName('MinLabelledPixels')
        self.SBMinPixels.SpinBox.setRange(1,1000)

        
        vlayout.addWidget(self.CBDitanceMap)
        vlayout.addWidget(self.CBIgnoreBackgroundTiles)
        vlayout.addWidget(self.SBDiscard)
        vlayout.addWidget(self.SBMinPixels)

        self.centralWidget.setLayout(vlayout)


class SegmentationSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SegmentationSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)

        self.CBIgnoreBackgroundTiles.stateChanged.connect(self.IgnoreBackgroundChanged)
        self.SBDiscard.SpinBox.valueChanged.connect(self.DiscardChanged)
        self.SBMinPixels.SpinBox.valueChanged.connect(self.MinPixelsChanged)

        self.CBDitanceMap.stateChanged.connect(self.ShapeSeparationChanged)
        
    def show(self):
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.ignoreBackground)
        self.SBDiscard.SpinBox.setValue(self.parent.parent.dl.augmentation.ignorecycles)
        self.SBMinPixels.SpinBox.setValue(self.parent.parent.dl.augmentation.minRequiredPixels)

        self.CBDitanceMap.setChecked(self.parent.parent.dl.Mode.useWeightedDistanceMap)
        super(SegmentationSettingsWindow, self).show()
        
    def ShapeSeparationChanged(self):
        self.parent.parent.dl.Mode.useWeightedDistanceMap = self.CBDitanceMap.isChecked()

    def IgnoreBackgroundChanged(self):
        self.parent.parent.dl.augmentation.ignoreBackground = self.CBIgnoreBackgroundTiles.isChecked()
        if self.CBIgnoreBackgroundTiles.isChecked():
            self.SBMinPixels.setEnabled(True)
            self.SBDiscard.setEnabled(True)
        else:
            self.SBMinPixels.setEnabled(False)
            self.SBDiscard.setEnabled(False)
        self.parent.parent.checkForTrainingWarnings()

    def MinPixelsChanged(self):
        self.parent.parent.dl.augmentation.minRequiredPixels = self.SBMinPixels.SpinBox.value()

    def DiscardChanged(self):
        self.parent.parent.dl.augmentation.ignorecycles = self.SBDiscard.SpinBox.value()