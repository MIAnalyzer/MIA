# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_utils import LabelledAdaptiveDoubleSpinBox, LabelledSpinBox
from ui.style import styleForm

class Window(object):
    def setupUi(self, Form):
        width = 300
        height= 380
        Form.setWindowTitle('Training Settings') 
        styleForm(Form)
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 3, 3, 3)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)

        self.CBMemory = QCheckBox("Load Full Dataset",self.centralWidget)
        self.CBMemory.setToolTip('Check to load full dataset into memory, uncheck to reload data on each iteration')
        

        self.vlayout.addWidget(self.CBMemory)
        self.vlayout.addWidget(self.LossGroup())
        self.vlayout.addWidget(self.TrainValGroup())
        self.vlayout.addWidget(self.LRScheduleGroup())

    def LossGroup(self):
        groupBox = QGroupBox("Loss, Metric, Optimizer")
        
        layout = QVBoxLayout(self.centralWidget)
        hlayout = QHBoxLayout(self.centralWidget)
        self.CBLoss = QComboBox(self.centralWidget)
        self.CBLoss.addItem("Focal Loss")
        self.CBLoss.setToolTip('Set loss function to be optimized during training')
        self.CBMetrics = QComboBox(self.centralWidget)
        self.CBMetrics.addItem("IoU")
        self.CBMetrics.setToolTip('Set metric to measure network perforance')
        self.CBOptimizer = QComboBox(self.centralWidget)
        self.CBOptimizer.addItem("Adam")
        self.CBOptimizer.setToolTip('Set optimizer to minimize loss function')
        hlayout.addWidget(self.CBLoss)
        hlayout.addWidget(self.CBMetrics)
        hlayout.addWidget(self.CBOptimizer)

        layout.addLayout(hlayout)
        self.CBDitanceMap = QCheckBox("Separate contours",self.centralWidget)
        self.CBDitanceMap.setToolTip('Check to separate adjacent contours')
        layout.addWidget(self.CBDitanceMap)

        groupBox.setLayout(layout)

        return groupBox

    def TrainValGroup(self):
        groupBox = QGroupBox("Train/Validation Split")
        
        hlayout = QHBoxLayout(self.centralWidget)
        self.STrainVal = QSlider(Qt.Horizontal, self.centralWidget)
        self.STrainVal.setMinimum(1)
        self.STrainVal.setMaximum(100)
        self.STrainVal.setValue(100)
        self.STrainVal.setToolTip('Split train and test set')
        self.STrainVal.setEnabled(False)
        self.LTrainVal = QLabel(self.centralWidget)
        self.LTrainVal.setText('100 % Train 0 % Val')
        hlayout.addWidget(self.STrainVal)
        hlayout.addWidget(self.LTrainVal)

        groupBox.setLayout(hlayout)

        return groupBox

    def LRScheduleGroup(self):
        groupBox = QGroupBox("Learning Rate Schedule")
        
        layout = QVBoxLayout(self.centralWidget)
        hlayout = QHBoxLayout(self.centralWidget)

        self.RBConstant = QRadioButton(self.centralWidget)
        self.RBConstant.setText("Constant")
        self.RBConstant.setToolTip('Toggle to set constant learning rate')
        self.RBConstant.setChecked(True)

        self.RBRlop = QRadioButton(self.centralWidget)
        self.RBRlop.setText("Plateau")
        self.RBRlop.setToolTip('Toggle to reduce learning rate on plateau')

        self.RBLRschedule = QRadioButton(self.centralWidget)
        self.RBLRschedule.setText("Schedule")
        self.RBLRschedule.setToolTip('Toggle to set learning rate schedule')

        hlayout.addWidget(self.RBConstant)
        hlayout.addWidget(self.RBRlop)
        hlayout.addWidget(self.RBLRschedule)
        layout.addLayout(hlayout)

        self.SBlr = LabelledAdaptiveDoubleSpinBox ('Learning Rate', self.centralWidget)
        self.SBlr.SpinBox.setRange(0.000001,1)
        self.SBlr.setToolTip('Set learning rate')
        self.SBlr.SpinBox.setDecimals(6)
        self.SBlr.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBlr)
        
        self.SBReduction = LabelledAdaptiveDoubleSpinBox ('Reduction Factor', self.centralWidget)
        self.SBReduction.SpinBox.setRange(0.000001,1)
        self.SBReduction.SpinBox.setDecimals(6)
        self.SBReduction.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBReduction)

        self.SBEveryx = LabelledSpinBox ('Reduce every x epochs', self.centralWidget)
        self.SBEveryx.SpinBox.setRange(1,1000)
        layout.addWidget(self.SBEveryx)

        groupBox.setLayout(layout)
        groupBox.setEnabled(False)
        return groupBox


class TrainSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TrainSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBMemory.setChecked(self.parent.parent.dl.TrainInMemory)
        self.CBDitanceMap.setChecked(self.parent.parent.dl.useWeightedDistanceMap)

        self.CBMemory.clicked.connect(self.InMemoryChanged)
        self.CBDitanceMap.clicked.connect(self.ShapeSeparationChanged)


    def InMemoryChanged(self):
        self.parent.parent.dl.TrainInMemory = self.CBMemory.isChecked()

    def ShapeSeparationChanged(self):
        self.parent.parent.dl.useWeightedDistanceMap = self.CBDitanceMap.isChecked()


