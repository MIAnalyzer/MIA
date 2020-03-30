# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:33:38 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class QAdaptiveDoubleSpinBox(QDoubleSpinBox):
    # implementation of QDoubleSpinBox with adaptive step size,
    # will be obsolete in future as QAbstractSpinBox::AdaptiveDecimalStepType is supported with qt 9.12 (not in python for now)
    def __init__(self, parent):
        super(QAdaptiveDoubleSpinBox, self).__init__(parent)
        self.epsilon = 1e-8
        
    def stepBy(self,steps):
        value = self.value()
        
        if steps < 0:
            self.setSingleStep(value/2)
        else:
            self.setSingleStep(value)

        QDoubleSpinBox.stepBy(self,steps)
        
    def textFromValue(self, value):
        return QLocale().toString(round(value, 6), 'f', QLocale.FloatingPointShortest)




class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Training Settings') 
        Form.setFixedSize(300, 150)
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(300, 150)
    
        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        # row 0
        self.CBModel = QComboBox(self.centralWidget)
        self.CBModel.addItem("Untrained Model")
        self.CBModel.addItem("Resnet50 Model")
        self.CBModel.setToolTip('Select model for training')
        self.vlayout.addWidget(self.CBModel)
        
        # row 1
        self.settings0layout = QHBoxLayout(self.centralWidget)
        self.SBClasses = QSpinBox(self.centralWidget)
        self.SBClasses.setRange(1,100)
        self.SBClasses.setToolTip('Number of classes')
        self.SBClasses.setEnabled(False)
        self.LClasses = QLabel(self.centralWidget)
        self.LClasses.setText('Classes')
        self.CBMono = QRadioButton(self.centralWidget)
        self.CBMono.setText("Mono")
        
        self.CBRGB = QRadioButton(self.centralWidget)
        self.CBRGB.setText("Color")
        self.CBRGB.setToolTip('Select if color images are used as input')
        self.settings0layout.addWidget(self.SBClasses)
        self.settings0layout.addWidget(self.LClasses)
        self.settings0layout.addWidget(self.CBRGB)
        self.settings0layout.addWidget(self.CBMono)
        self.vlayout.addLayout(self.settings0layout)
        
        # row 2
        self.settings1layout = QHBoxLayout(self.centralWidget)
        self.SBBatchSize = QSpinBox(self.centralWidget)
        self.SBBatchSize.setToolTip('Set batch size')
        self.SBBatchSize.setRange(1,2048)
        
        self.LBatchSize = QLabel(self.centralWidget)
        self.LBatchSize.setText('Batch Size')
        self.SBEpochs = QSpinBox(self.centralWidget)
        self.SBEpochs.setToolTip('Set number of epochs')
        self.SBEpochs.setRange(1,10000)
        
        self.LEpochs = QLabel(self.centralWidget)
        self.LEpochs.setText('Epochs')
        self.settings1layout.addWidget(self.SBBatchSize)
        self.settings1layout.addWidget(self.LBatchSize)
        self.settings1layout.addWidget(self.SBEpochs)
        self.settings1layout.addWidget(self.LEpochs)
        self.vlayout.addLayout(self.settings1layout)
        
        # row 3
        self.settings2layout = QHBoxLayout(self.centralWidget)
        self.SBLearningRate = QAdaptiveDoubleSpinBox (self.centralWidget)
        self.SBLearningRate.setRange(0.000001,1)
        self.SBLearningRate.setToolTip('Set learning rate')
        self.SBLearningRate.setDecimals(6)
        self.SBLearningRate.setSingleStep(0.1)
        
        self.LLearningRate = QLabel(self.centralWidget)
        self.LLearningRate.setText('Learning Rate')
        self.SBScaleFactor = QDoubleSpinBox(self.centralWidget)
        self.SBScaleFactor.setRange(0.1,1)
        self.SBScaleFactor.setToolTip('Set image reduction factor')
        self.SBScaleFactor.setSingleStep(0.1)
        self.SBScaleFactor.setDecimals(1)
        
        self.LScaleFactor = QLabel(self.centralWidget)
        self.LScaleFactor.setText('Down Scaling')
        self.settings2layout.addWidget(self.SBLearningRate)
        self.settings2layout.addWidget(self.LLearningRate)
        self.settings2layout.addWidget(self.SBScaleFactor)
        self.settings2layout.addWidget(self.LScaleFactor)
        self.vlayout.addLayout(self.settings2layout)
        
        # row 4
        self.settings3layout = QHBoxLayout(self.centralWidget)
        self.BSettings = QPushButton(self.centralWidget)
        self.BSettings.setText('Settings')
        self.BSettings.setToolTip('Open extended training settings')
        self.BSettings.setFlat(True)
        self.BSettings.setIcon(QIcon('icons/augmentation.png'))
        self.BSettings.setStyleSheet('text-align:left')
        self.settings3layout.addWidget(self.BSettings)
        self.vlayout.addLayout(self.settings3layout)
        self.BAugmentation = QPushButton(self.centralWidget)
        self.BAugmentation.setToolTip('Open image augmentation settings')
        self.BAugmentation.setText('Augmentations')
        self.BAugmentation.setFlat(True)
        self.BAugmentation.setIcon(QIcon('icons/augmentation.png'))
        self.BAugmentation.setStyleSheet('text-align:left')
        self.settings3layout.addWidget(self.BAugmentation)
        self.vlayout.addLayout(self.settings3layout)
        
        # row 5
        self.BTrain = QPushButton(self.centralWidget)
        self.BTrain.setIcon(QIcon('icons/train.png'))
        self.BTrain.setFlat(True)
        self.BTrain.setToolTip('Start training with selected parameters')
        self.BTrain.setText('Start Training')
        self.BTrain.setStyleSheet('border:1px solid black ')
        self.vlayout.addWidget(self.BTrain)  
        
        
    
    
    
class TrainingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TrainingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)   
        
        self.SBLearningRate.setValue(self.parent.dl.learning_rate)
        self.CBMono.setChecked(True)
        self.SBEpochs.setValue(self.parent.dl.epochs)
        self.SBScaleFactor.setValue(self.parent.dl.ImageScaleFactor)
        self.SBBatchSize.setValue(self.parent.dl.batch_size)
        
        
        self.SBBatchSize.valueChanged.connect(self.BatchSizeChanged)
        self.SBEpochs.valueChanged.connect(self.EpochsChanged)
        self.SBLearningRate.valueChanged.connect(self.LRChanged)
        self.SBScaleFactor.valueChanged.connect(self.ScaleFactorChanged)
        self.CBModel.currentIndexChanged.connect(self.ModelType)
        #self.BSettings
        #self.BAugmentation
        self.BTrain.clicked.connect(self.Train)
        
        
    def ModelType(self):
        self.parent.dl.ModelType = self.CBModel.currentIndex()
        
    def BatchSizeChanged(self):
        self.parent.dl.batch_size = self.SBBatchSize.value()
        
    def EpochsChanged(self):
        self.parent.dl.epochs = self.SBEpochs.value()
        
    def LRChanged(self):
        self.parent.dl.learning_rate = self.SBLearningRate.value()
        
    def ScaleFactorChanged(self):
        self.parent.dl.ImageScaleFactor = self.SBScaleFactor.value()
  
        
    def Train(self):
        if not self.parent.dl.initialized:
            if not self.parent.dl.initModel(self.parent.NumOfClasses(), self.CBMono.isChecked()):
               self.parent.PopupWarning('Cannot train model (out of recources?)') 
               return
        if not self.parent.dl.Train(self.parent.trainImagespath,self.parent.trainImageLabelspath):
            self.parent.PopupWarning('Cannot train model (out of recources?)')
        self.hide()
        
        