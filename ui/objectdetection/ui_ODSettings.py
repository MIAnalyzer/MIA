# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:32:23 2020

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
        height= 140
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
        self.SObjectSize.setObjectName('CountingObjectSize')
        self.SObjectSize.setToolTip('Use to adapt object size and distribution. Use smaller values for small and densely distributed objects')
        self.LObjSize = QLabel(self.centralWidget)
        self.LObjSize.setText('Object Size')
        hlayout.addWidget(self.SObjectSize)
        hlayout.addWidget(self.LObjSize)
        
        h2layout = QHBoxLayout(self.centralWidget)
        self.SObjectIntensity =  QSlider(Qt.Horizontal, self.centralWidget)
        self.SObjectIntensity.setMinimum(1)
        self.SObjectIntensity.setMaximum(500)
        self.SObjectIntensity.setObjectName('CountingObjectIntensity')
        self.SObjectIntensity.setToolTip('Use to set object peak intensity')
        self.LObjIntensity = QLabel(self.centralWidget)
        self.LObjIntensity.setText('Object Intensity')

        self.CBIgnoreBackgroundTiles = QCheckBox("Prefer labelled parts",self.centralWidget)
        self.CBIgnoreBackgroundTiles.setToolTip('When checked image parts which contain only background are ignored during training')
        self.CBIgnoreBackgroundTiles.setObjectName('IgnoreBackground')

        self.SBDiscard = LabelledSpinBox ('Discard up to x background tiles', self.centralWidget)
        self.SBDiscard.setObjectName('DiscardBGTiles')
        self.SBDiscard.SpinBox.setRange(1,20)

        self.SBMinPixels = LabelledSpinBox ('Minimum required labelled pixels', self.centralWidget)
        self.SBMinPixels.setObjectName('MinLabelledPixels')
        self.SBMinPixels.SpinBox.setRange(1,1000)


        
        h2layout.addWidget(self.SObjectIntensity)
        h2layout.addWidget(self.LObjIntensity)


        vlayout.addLayout(hlayout)
        vlayout.addLayout(h2layout)
        vlayout.addWidget(self.CBIgnoreBackgroundTiles)
        vlayout.addWidget(self.SBDiscard)
        vlayout.addWidget(self.SBMinPixels)



        self.centralWidget.setLayout(vlayout)

class ObjectDetectionSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ObjectDetectionSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.SObjectSize.valueChanged.connect(self.ObjectSizeChanged)
        self.SObjectIntensity.valueChanged.connect(self.ObjectIntensityChanged)

        
        self.CBIgnoreBackgroundTiles.stateChanged.connect(self.IgnoreBackgroundChanged)
        self.SBDiscard.SpinBox.valueChanged.connect(self.DiscardChanged)
        self.SBMinPixels.SpinBox.valueChanged.connect(self.MinPixelsChanged)
        
    def show(self):
        self.SObjectSize.setValue(self.parent.parent.dl.Mode.kernel_stdev)
        self.SObjectIntensity.setValue(self.parent.parent.dl.Mode.peak_val)
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.ignoreBackground)
        self.SBDiscard.SpinBox.setValue(self.parent.parent.dl.augmentation.ignorecycles)
        self.SBMinPixels.SpinBox.setValue(self.parent.parent.dl.augmentation.minRequiredPixels)
        super(ObjectDetectionSettingsWindow, self).show()
        
    def ObjectSizeChanged(self):
        self.parent.parent.dl.Mode.kernel_stdev = self.SObjectSize.value()
        
    def ObjectIntensityChanged(self):
        self.parent.parent.dl.Mode.peak_val = self.SObjectIntensity.value()
        

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
        
        
        