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

        Form.setWindowTitle('Segmentation Settings') 
        styleForm(Form)  

        self.centralWidget = QWidget(Form)
        
        vlayout = QVBoxLayout(self.centralWidget)
        vlayout.setContentsMargins(3, 3, 3, 3)
        vlayout.setSpacing(2)
        
        self.CBDitanceMap = QCheckBox("Separate contours",self.centralWidget)
        self.CBDitanceMap.setObjectName('nn_CreateWeightedDistanceMap')
        self.CBDitanceMap.setToolTip('Check to separate adjacent contours')

        self.CBIgnoreBackgroundTiles = QCheckBox("Prefer labelled parts",self.centralWidget)
        self.CBIgnoreBackgroundTiles.setToolTip('When checked image parts which contain only background are ignored during training')
        self.CBIgnoreBackgroundTiles.setObjectName('nn_IgnoreBackground_seg')

        self.SBDiscard = LabelledSpinBox ('Discard up to x background tiles', self.centralWidget)
        self.SBDiscard.setObjectName('nn_DiscardBGTiles_seg')
        self.SBDiscard.SpinBox.setRange(1,20)

        self.SBMinPixels = LabelledSpinBox ('Minimum required labelled pixels', self.centralWidget)
        self.SBMinPixels.setObjectName('nn_MinLabelledPixels_seg')
        self.SBMinPixels.SpinBox.setRange(1,1000)

        
        vlayout.addWidget(self.CBDitanceMap)
        vlayout.addWidget(self.CBIgnoreBackgroundTiles)
        vlayout.addWidget(self.SBDiscard)
        vlayout.addWidget(self.SBMinPixels)

        self.centralWidget.setLayout(vlayout)

        self.centralWidget.setFixedSize(vlayout.sizeHint())
        Form.setFixedSize(vlayout.sizeHint())
                


class SegmentationSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SegmentationSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)

        self.CBIgnoreBackgroundTiles.stateChanged.connect(self.IgnoreBackgroundChanged)
        self.SBDiscard.SpinBox.valueChanged.connect(self.DiscardChanged)
        self.SBMinPixels.SpinBox.valueChanged.connect(self.MinPixelsChanged)

        self.CBDitanceMap.stateChanged.connect(self.ShapeSeparationChanged)
        
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.seg_ignoreBackground)
        self.CBDitanceMap.setChecked(self.parent.parent.dl.seg_useWeightedDistanceMap)
        
    def show(self):
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.seg_ignoreBackground)
        self.SBDiscard.SpinBox.setValue(self.parent.parent.dl.augmentation.seg_ignorecycles)
        self.SBMinPixels.SpinBox.setValue(self.parent.parent.dl.augmentation.seg_minRequiredPixels)

        self.CBDitanceMap.setChecked(self.parent.parent.dl.seg_useWeightedDistanceMap)
        super(SegmentationSettingsWindow, self).show()
        
    def ShapeSeparationChanged(self):
        self.parent.parent.dl.seg_useWeightedDistanceMap = self.CBDitanceMap.isChecked()

    def IgnoreBackgroundChanged(self):
        self.parent.parent.dl.augmentation.seg_ignoreBackground = self.CBIgnoreBackgroundTiles.isChecked()
        if self.CBIgnoreBackgroundTiles.isChecked():
            self.SBMinPixels.setEnabled(True)
            self.SBDiscard.setEnabled(True)
        else:
            self.SBMinPixels.setEnabled(False)
            self.SBDiscard.setEnabled(False)

    def MinPixelsChanged(self):
        self.parent.parent.dl.augmentation.seg_minRequiredPixels = self.SBMinPixels.SpinBox.value()

    def DiscardChanged(self):
        self.parent.parent.dl.augmentation.seg_ignorecycles = self.SBDiscard.SpinBox.value()