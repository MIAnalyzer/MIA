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
        Form.setWindowTitle('Object Counting Settings') 
        styleForm(Form)
        
        self.centralWidget = QWidget(Form)
         
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(3, 3, 3, 3)
        vlayout.setSpacing(2)
        

        hlayout = QHBoxLayout()
        self.SObjectSize =  QSlider(Qt.Horizontal, self.centralWidget)
        self.SObjectSize.setMinimum(1)
        self.SObjectSize.setMaximum(255)
        self.SObjectSize.setObjectName('nn_CountingObjectSize')
        self.SObjectSize.setToolTip('Use to adapt object size and distribution. Use smaller values for small and densely distributed objects')
        self.LObjSize = QLabel(self.centralWidget)
        self.LObjSize.setText('Object Size')
        hlayout.addWidget(self.SObjectSize)
        hlayout.addWidget(self.LObjSize)
        
        h2layout = QHBoxLayout()
        self.SObjectIntensity =  QSlider(Qt.Horizontal, self.centralWidget)
        self.SObjectIntensity.setMinimum(1)
        self.SObjectIntensity.setMaximum(500)
        self.SObjectIntensity.setObjectName('nn_CountingObjectIntensity')
        self.SObjectIntensity.setToolTip('Use to set object peak intensity')
        self.LObjIntensity = QLabel(self.centralWidget)
        self.LObjIntensity.setText('Object Intensity')

        self.CBIgnoreBackgroundTiles = QCheckBox("Prefer labelled parts",self.centralWidget)
        self.CBIgnoreBackgroundTiles.setToolTip('When checked image parts which contain only background are ignored during training')
        self.CBIgnoreBackgroundTiles.setObjectName('nn_IgnoreBackground_od')

        self.SBDiscard = LabelledSpinBox ('Discard up to x background tiles', self.centralWidget)
        self.SBDiscard.setObjectName('nn_DiscardBGTiles_od')
        self.SBDiscard.SpinBox.setRange(1,20)

        self.SBMinPixels = LabelledSpinBox ('Minimum required labelled pixels', self.centralWidget)
        self.SBMinPixels.setObjectName('nn_MinLabelledPixels_od')
        self.SBMinPixels.SpinBox.setRange(1,1000)


        
        h2layout.addWidget(self.SObjectIntensity)
        h2layout.addWidget(self.LObjIntensity)


        vlayout.addLayout(hlayout)
        vlayout.addLayout(h2layout)
        vlayout.addWidget(self.CBIgnoreBackgroundTiles)
        vlayout.addWidget(self.SBDiscard)
        vlayout.addWidget(self.SBMinPixels)


        self.centralWidget.setLayout(vlayout)
        self.centralWidget.setFixedSize(vlayout.sizeHint())
        Form.setFixedSize(vlayout.sizeHint())
                


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
        
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.od_ignoreBackground)
        
    def show(self):
        self.SObjectSize.setValue(self.parent.parent.dl.od_kernel_size)
        self.SObjectIntensity.setValue(self.parent.parent.dl.od_peak_val)
        self.CBIgnoreBackgroundTiles.setChecked(self.parent.parent.dl.augmentation.od_ignoreBackground)
        self.SBDiscard.SpinBox.setValue(self.parent.parent.dl.augmentation.od_ignorecycles)
        self.SBMinPixels.SpinBox.setValue(self.parent.parent.dl.augmentation.od_minRequiredPixels)
        super(ObjectDetectionSettingsWindow, self).show()
        
    def ObjectSizeChanged(self):
        self.parent.parent.dl.od_kernel_size = self.SObjectSize.value()
        
    def ObjectIntensityChanged(self):
        self.parent.parent.dl.od_peak_val = self.SObjectIntensity.value()

    def IgnoreBackgroundChanged(self):
        self.parent.parent.dl.augmentation.od_ignoreBackground = self.CBIgnoreBackgroundTiles.isChecked()
        
        if self.CBIgnoreBackgroundTiles.isChecked():
            self.SBMinPixels.setEnabled(True)
            self.SBDiscard.setEnabled(True)
        else:
            self.SBMinPixels.setEnabled(False)
            self.SBDiscard.setEnabled(False)

    def MinPixelsChanged(self):
        self.parent.parent.dl.augmentation.od_minRequiredPixels = self.SBMinPixels.SpinBox.value()

    def DiscardChanged(self):
        self.parent.parent.dl.augmentation.od_ignorecycles = self.SBDiscard.SpinBox.value()
        
        
        