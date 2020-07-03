# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:33:38 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_Augment import AugmentWindow
from ui.ui_TrainSettings import TrainSettingsWindow
from ui.ui_utils import LabelledAdaptiveDoubleSpinBox, LabelledSpinBox, LabelledDoubleSpinBox, DCDButton
from ui.style import styleForm


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Training Settings') 
        width = 350
        height= 250

        Form.setFixedSize(width, height)
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
    
        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        # row 0
        hlayout = QHBoxLayout(self.centralWidget)
        self.CBModel = QComboBox(self.centralWidget)
        self.CBModel.addItem("Untrained Model")
        self.CBModel.addItem("Resnet50 Model")
        self.CBModel.setToolTip('Select model for training')
        self.LModel = QLabel(self.centralWidget)
        self.LModel.setText('Deep Learning Model')
        hlayout.addWidget(self.CBModel)
        hlayout.addWidget(self.LModel)
        
        self.vlayout.addLayout(hlayout)
        
        self.Parameter = self.ParameterGroup()
        self.vlayout.addWidget(self.Parameter)

        # row 4
        self.settings3layout = QHBoxLayout(self.centralWidget)
        self.BSettings = DCDButton(self.centralWidget, 'Settings')
        self.BSettings.setToolTip('Open extended training settings')
        self.BSettings.setIcon(QIcon('icons/augmentation.png'))
        self.settings3layout.addWidget(self.BSettings)
        self.vlayout.addLayout(self.settings3layout)
        self.BAugmentation = DCDButton(self.centralWidget, 'Augmentations')
        self.BAugmentation.setToolTip('Open image augmentation settings')
        self.BAugmentation.setIcon(QIcon('icons/augmentation.png'))
        self.settings3layout.addWidget(self.BAugmentation)
        self.vlayout.addLayout(self.settings3layout)
        
        # row 5
        self.BTrain = DCDButton(self.centralWidget, 'Start Training')
        self.BTrain.setIcon(QIcon('icons/train.png'))
        self.BTrain.setToolTip('Start training with selected parameters')
        self.vlayout.addWidget(self.BTrain)  

    def ParameterGroup(self):
        groupBox = QGroupBox("Training Parameter")
        layout = QVBoxLayout(self.centralWidget)
        # row 1
        self.settings0layout = QHBoxLayout(self.centralWidget)
        self.SBClasses = LabelledSpinBox('Classes', self.centralWidget)
        self.SBClasses.SpinBox.setRange(1,100)
        self.SBClasses.setToolTip('Number of classes')
        self.SBClasses.SpinBox.setEnabled(False)

        self.CBMono = QRadioButton(self.centralWidget)
        self.CBMono.setText("Mono")
        self.CBMono.setToolTip('Select if only monochromatic images are used as input')
        
        self.CBRGB = QRadioButton(self.centralWidget)
        self.CBRGB.setText("Color")
        self.CBRGB.setToolTip('Select if color images are used as input')
        self.settings0layout.addWidget(self.SBClasses,stretch = 2)
        self.settings0layout.addWidget(self.CBRGB, stretch = 1)
        self.settings0layout.addWidget(self.CBMono, stretch = 1)
        layout.addLayout(self.settings0layout)
        
        # row 2
        self.settings1layout = QHBoxLayout(self.centralWidget)
        self.SBBatchSize = LabelledSpinBox('Batch Size',self.centralWidget)
        self.SBBatchSize.setToolTip('Set batch size')
        self.SBBatchSize.SpinBox.setRange(1,2048)

        self.SBEpochs = LabelledSpinBox('Epochs',self.centralWidget)
        self.SBEpochs.setToolTip('Set number of epochs')
        self.SBEpochs.SpinBox.setRange(1,10000)
        self.settings1layout.addWidget(self.SBBatchSize)
        self.settings1layout.addWidget(self.SBEpochs)

        layout.addLayout(self.settings1layout)
        
        # row 3
        self.settings2layout = QHBoxLayout(self.centralWidget)
        self.SBLearningRate = LabelledAdaptiveDoubleSpinBox ('Learning Rate',self.centralWidget)
        self.SBLearningRate.SpinBox.setRange(0.000001,1)
        self.SBLearningRate.setToolTip('Set learning rate')
        self.SBLearningRate.SpinBox.setDecimals(6)
        self.SBLearningRate.SpinBox.setSingleStep(0.1)

        self.SBScaleFactor = LabelledDoubleSpinBox('Down Scaling',self.centralWidget)
        self.SBScaleFactor.SpinBox.setRange(0.1,1)
        self.SBScaleFactor.setToolTip('Set image reduction factor')
        self.SBScaleFactor.SpinBox.setSingleStep(0.1)
        self.SBScaleFactor.SpinBox.setDecimals(1)

        self.settings2layout.addWidget(self.SBLearningRate)
        self.settings2layout.addWidget(self.SBScaleFactor)
        layout.addLayout(self.settings2layout)
        groupBox.setLayout(layout)
        return groupBox
        

class TrainingWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TrainingWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)   
        
        self.SBLearningRate.SpinBox.setMinimum(self.parent.dl.lrschedule.minlr)
        self.SBLearningRate.SpinBox.setValue(self.parent.dl.learning_rate)
        self.CBMono.setChecked(True)
        self.SBEpochs.SpinBox.setValue(self.parent.dl.epochs)
        self.SBScaleFactor.SpinBox.setValue(self.parent.dl.ImageScaleFactor)
        self.SBBatchSize.SpinBox.setValue(self.parent.dl.batch_size)
        
        
        self.SBBatchSize.SpinBox.valueChanged.connect(self.BatchSizeChanged)
        self.SBEpochs.SpinBox.valueChanged.connect(self.EpochsChanged)
        self.SBLearningRate.SpinBox.valueChanged.connect(self.LRChanged)
        self.SBScaleFactor.SpinBox.valueChanged.connect(self.ScaleFactorChanged)
        self.CBModel.currentIndexChanged.connect(self.ModelType)
        
        self.settings_form = TrainSettingsWindow(self)
        self.augmentation_form = AugmentWindow(self.parent)

        self.BSettings.clicked.connect(self.settings_form.show)
        self.BAugmentation.clicked.connect(self.augmentation_form.show)
        self.BTrain.clicked.connect(self.Train)
        
    def show(self):
        self.toggleParameterStatus()
        super(TrainingWindow, self).show()
        
    def toggleParameterStatus(self):
        if not self.parent.dl.initialized():
            self.CBMono.setEnabled(True)
            self.CBRGB.setEnabled(True)
            self.CBModel.setEnabled(True)
        else:
            self.CBMono.setEnabled(False)
            self.CBRGB.setEnabled(False)
            self.CBModel.setEnabled(False)
            if self.parent.dl.MonoChrome():
                self.CBMono.setChecked(True)
            else:
                self.CBRGB.setChecked(True) 
        
    def hide(self):
        self.settings_form.hide()
        self.augmentation_form.hide()
        super(TrainingWindow, self).hide()
        
    def closeEvent(self, event):       
        self.settings_form.hide()
        self.augmentation_form.hide()
        super(TrainingWindow, self).closeEvent(event)
        
    def ModelType(self):
        self.parent.dl.ModelType = self.CBModel.currentIndex()
        
    def BatchSizeChanged(self):
        self.parent.dl.batch_size = self.SBBatchSize.SpinBox.value()
        
    def EpochsChanged(self):
        self.parent.dl.epochs = self.SBEpochs.SpinBox.value()
        
    def LRChanged(self):
        self.parent.dl.learning_rate = self.SBLearningRate.SpinBox.value()
        self.settings_form.silentlyUpdateLR(self.SBLearningRate.SpinBox.value())      
        
    def ScaleFactorChanged(self):
        self.parent.dl.ImageScaleFactor = self.SBScaleFactor.SpinBox.value()
        self.parent.settings_form.silentlyUpdateScale(self.SBScaleFactor.SpinBox.value())
  
    def Train(self):
        self.parent.startTraining()
        
    def toggleTrainStatus(self, training):
        self.BTrain.setEnabled(not training)
        self.setEnabled(not training)
        self.settings_form.setEnabled(not training)
        self.augmentation_form.setEnabled(not training)


        
        