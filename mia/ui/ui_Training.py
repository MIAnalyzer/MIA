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
from ui.segmentation.ui_SegSettings import SegmentationSettingsWindow
from ui.objectdetection.ui_ODSettings import ObjectDetectionSettingsWindow
from ui.ui_utils import LabelledAdaptiveDoubleSpinBox, LabelledSpinBox, LabelledDoubleSpinBox, DCDButton
from ui.style import styleForm
from dl.method.mode import dlMode
from dl.data.imagedata import dlChannels


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Neural Network Training') 

        styleForm(Form)
        self.centralWidget = QWidget(Form)
    
        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        # row 0
        self.Model = self.ModelGroup()
        self.vlayout.addWidget(self.Model)
        
        self.Parameter = self.ParameterGroup()
        self.vlayout.addWidget(self.Parameter)

        # row 4
        self.settings3layout = QHBoxLayout()
        self.BModeSettings = DCDButton(self.centralWidget, 'Extended')
        self.BModeSettings.setToolTip('Open extended settings')
        self.BModeSettings.setIcon(QIcon('icons/settings.png'))
        self.settings3layout.addWidget(self.BModeSettings)
        self.BSettings = DCDButton(self.centralWidget, 'Settings')
        self.BSettings.setToolTip('Open training settings')
        self.BSettings.setIcon(QIcon('icons/settings.png'))
        self.settings3layout.addWidget(self.BSettings)
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


        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())


    def ModelGroup(self):
        groupBox = QGroupBox("Deep Learning Model")
        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        self.CBArchitecture = QComboBox(self.centralWidget)
        self.CBArchitecture.setObjectName('nn_Architecture')
        self.CBArchitecture.setToolTip('Select model architecture for training')
        self.LArchitecture = QLabel(self.centralWidget)
        self.LArchitecture.setText('Deep Learning Architecture')
        hlayout.addWidget(self.CBArchitecture)
        hlayout.addWidget(self.LArchitecture)
     
        layout.addLayout(hlayout)

        h2layout = QHBoxLayout()
        self.CBBackbone = QComboBox(self.centralWidget)
        self.CBBackbone.setObjectName('nn_Backbone')
        self.CBBackbone.setToolTip('Select model Backbone')
        self.LBackbone = QLabel(self.centralWidget)
        self.LBackbone.setText('Model Backbone')
        h2layout.addWidget(self.CBBackbone)
        h2layout.addWidget(self.LBackbone)
     
        layout.addLayout(h2layout)

        self.CBPretrained = QCheckBox("Pretrained Net",self.centralWidget)
        self.CBPretrained.setObjectName('nn_preTrained')
        self.CBPretrained.setToolTip('Select to use an imagenet-pretrained model')
        layout.addWidget(self.CBPretrained)
        groupBox.setLayout(layout)
        return groupBox


    def ParameterGroup(self):
        groupBox = QGroupBox("Training Parameter")
        layout = QVBoxLayout()
        # row 1
        self.settings0layout = QHBoxLayout()
        self.CBMono = QRadioButton(self.centralWidget)
        self.CBMono.setText("Mono")
        self.CBMono.setObjectName('nn_MonochromaticInput')
        self.CBMono.setToolTip('Select to use monochromatic images as input')
        
        self.CBRGB = QRadioButton(self.centralWidget)
        self.CBRGB.setText("Color")
        self.CBRGB.setObjectName('nn_RGBInput')
        self.CBRGB.setToolTip('Select to use color images as input')
        
        self.CBnChan = QRadioButton(self.centralWidget)
        self.CBnChan.setText("nChan")
        self.CBnChan.setObjectName('nn_nChanInput')
        self.CBnChan.setToolTip('Select to use n-channel images as input')
        
        
        self.settings0layout.addWidget(self.CBRGB)
        self.settings0layout.addWidget(self.CBMono)
        self.settings0layout.addWidget(self.CBnChan)
        layout.addLayout(self.settings0layout)
        
        
        # row 1.5
        self.settings15layout = QHBoxLayout()
        
        self.SBClasses = LabelledSpinBox('Classes', self.centralWidget)
        self.SBClasses.setObjectName('nn_NumOfClasses')
        self.SBClasses.SpinBox.setRange(1,100)
        self.SBClasses.setToolTip('Number of classes')
        self.SBClasses.SpinBox.setEnabled(False)
        
        self.SBnumChan = LabelledSpinBox('Input Channels',self.centralWidget)
        self.SBnumChan.setToolTip('Set number of input channels')
        self.SBnumChan.setObjectName('nn_InputChannels')
        self.SBnumChan.SpinBox.setRange(1,56)
        
        self.settings15layout.addWidget(self.SBClasses, stretch = 1)
        self.settings15layout.addWidget(self.SBnumChan, stretch = 1)
        layout.addLayout(self.settings15layout)

        
        # row 2
        self.settings1layout = QHBoxLayout()
        self.SBBatchSize = LabelledSpinBox('Batch Size',self.centralWidget)
        self.SBBatchSize.setToolTip('Set batch size')
        self.SBBatchSize.setObjectName('nn_BatchSize')
        self.SBBatchSize.SpinBox.setRange(1,2048)

        self.SBEpochs = LabelledSpinBox('Epochs',self.centralWidget)
        self.SBEpochs.setToolTip('Set number of epochs')
        self.SBEpochs.setObjectName('nn_Epochs')
        self.SBEpochs.SpinBox.setRange(1,10000)
        self.settings1layout.addWidget(self.SBBatchSize, stretch = 1)
        self.settings1layout.addWidget(self.SBEpochs, stretch = 1)

        layout.addLayout(self.settings1layout)
        
        # row 3
        self.settings2layout = QHBoxLayout()
        self.SBLearningRate = LabelledAdaptiveDoubleSpinBox ('Learning Rate',self.centralWidget)
        self.SBLearningRate.SpinBox.setRange(0.000001,1)
        self.SBLearningRate.setToolTip('Set learning rate')
        self.SBLearningRate.SpinBox.setDecimals(6)
        self.SBLearningRate.SpinBox.setSingleStep(0.1)

        self.SBScaleFactor = LabelledDoubleSpinBox('Down Scaling',self.centralWidget)
        self.SBScaleFactor.SpinBox.setRange(0.1,1)
        self.SBScaleFactor.setObjectName('nn_ImageDownScaling')
        self.SBScaleFactor.setToolTip('Set image reduction factor')
        self.SBScaleFactor.SpinBox.setSingleStep(0.1)
        self.SBScaleFactor.SpinBox.setDecimals(1)

        self.settings2layout.addWidget(self.SBLearningRate, stretch = 1)
        self.settings2layout.addWidget(self.SBScaleFactor, stretch = 1)
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
        self.SBEpochs.SpinBox.setValue(self.parent.dl.epochs)
        self.SBScaleFactor.SpinBox.setValue(self.parent.dl.ImageScaleFactor)
        self.SBBatchSize.SpinBox.setValue(self.parent.dl.batch_size)
        self.SBnumChan.SpinBox.setValue(self.parent.dl.data.numChannels)
        
        
        
        self.SBBatchSize.SpinBox.valueChanged.connect(self.BatchSizeChanged)
        self.SBEpochs.SpinBox.valueChanged.connect(self.EpochsChanged)
        self.SBLearningRate.SpinBox.valueChanged.connect(self.LRChanged)
        self.SBScaleFactor.SpinBox.valueChanged.connect(self.ScaleFactorChanged)
        self.CBArchitecture.currentIndexChanged.connect(self.ArchitectureChanged)
        self.CBBackbone.currentIndexChanged.connect(self.BackBoneChanged)
        self.CBPretrained.setChecked(self.parent.dl.Mode.pretrained)
        self.CBPretrained.stateChanged.connect(self.UsePretrained)
        self.CBMono.toggled.connect(self.ChannelsChanged)
        self.CBRGB.toggled.connect(self.ChannelsChanged)
        self.CBnChan.toggled.connect(self.ChannelsChanged)
        self.SBnumChan.SpinBox.valueChanged.connect(self.inputChannelsChanged)
        

        self.setModelOptions()

        self.settings_form = TrainSettingsWindow(self)
        self.augmentation_form = AugmentWindow(self.parent)
        # training mode settings
        self.segmentationssettings_form = SegmentationSettingsWindow(self)
        self.objectdetectionsettings_form = ObjectDetectionSettingsWindow(self)
        

        self.BSettings.clicked.connect(self.settings_form.show)
        self.BAugmentation.clicked.connect(self.augmentation_form.show)
        self.BModeSettings.clicked.connect(self.showModeSettings)
        self.BTrain.clicked.connect(self.Train)
        self.changeStackLabelOption()
        
    def show(self):
        self.setParameterStatus()
        self.ModeChanged()
        super(TrainingWindow, self).show()
        
    def setParameterStatus(self):
        if not self.parent.dl.initialized:
            self.CBMono.setEnabled(True)
            self.CBRGB.setEnabled(True)
            self.CBnChan.setEnabled(not self.parent.separateStackLabels)
            self.CBArchitecture.setEnabled(True)
            self.CBBackbone.setEnabled(True)
            self.CBPretrained.setEnabled(True)
        else:
            self.CBMono.setEnabled(False)
            self.CBRGB.setEnabled(False)
            self.CBnChan.setEnabled(False)
            self.CBArchitecture.setEnabled(False)
            self.CBBackbone.setEnabled(False)
            self.CBPretrained.setEnabled(False)
        self.getChannels()

        
    def changeStackLabelOption(self):
        self.CBnChan.setEnabled(not self.parent.separateStackLabels)
        if self.CBnChan.isChecked():
            self.CBMono.setChecked(True)
            

    def getChannels(self):
        if self.parent.dl.data.channels == dlChannels.Mono:
            self.CBMono.setChecked(True)
        elif self.parent.dl.data.channels == dlChannels.RGB:
            self.CBRGB.setChecked(True) 
        elif self.parent.dl.data.channels == dlChannels.nChan:
            self.CBnChan.setChecked(True) 
            
        self.SBnumChan.SpinBox.setValue(self.parent.dl.data.numChannels)
        
    def ChannelsChanged(self):
        self.SBnumChan.setEnabled(False)
        if self.CBMono.isChecked():
            self.parent.dl.data.channels = dlChannels.Mono
        elif self.CBRGB.isChecked():
            self.parent.dl.data.channels = dlChannels.RGB
        elif self.CBnChan.isChecked():
            self.parent.dl.data.channels = dlChannels.nChan
            self.SBnumChan.setEnabled(True)
        self.SBnumChan.SpinBox.setValue(self.parent.dl.data.numChannels)
            
            
    def inputChannelsChanged(self):
        if self.parent.dl.data.channels == dlChannels.nChan:
            self.parent.dl.data.nchannels = self.SBnumChan.SpinBox.value()
        
    def hide(self):
        self.closeWindows()
        super(TrainingWindow, self).hide()
        
    def closeEvent(self, event):       
        self.closeWindows()
        super(TrainingWindow, self).closeEvent(event)
        
    def closeWindows(self):
        self.settings_form.hide()
        self.augmentation_form.hide()
        self.segmentationssettings_form.hide()
        self.objectdetectionsettings_form.hide()
        
    def showModeSettings(self):
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.segmentationssettings_form.show()
        elif self.parent.LearningMode() == dlMode.Object_Counting:
            self.objectdetectionsettings_form.show()    
    
    def ModeChanged(self):
        self.settings_form.updateLossesAndMetrics()
        self.closeWindows()
        self.setModelOptions()
        self.UsePretrained()
        self.BModeSettings.setEnabled(True)
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.BModeSettings.setToolTip('Open segmentation settings')
            self.BModeSettings.setText('Segmentation')
        elif self.parent.LearningMode() == dlMode.Object_Counting:
            self.BModeSettings.setToolTip('Open object detection settings')
            self.BModeSettings.setText('Detection')
        elif self.parent.LearningMode() == dlMode.Classification:
            self.BModeSettings.setToolTip('Open classification settings')
            self.BModeSettings.setText('Classification')
            self.BModeSettings.setEnabled(False)
            

    def setModelOptions(self):
        self.setArchitectureOptions()
        self.setBackBoneOptions()
        
    def setArchitectureOptions(self):
        defaultarchitecture= self.parent.dl.Mode.architecture
        defaultbackbone = self.parent.dl.Mode.backbone
        self.CBArchitecture.clear()
        for arch in self.parent.dl.Mode.getArchitectures():
            self.CBArchitecture.addItem(arch)   
        self.CBArchitecture.setCurrentIndex(self.CBArchitecture.findText(defaultarchitecture))
        self.CBBackbone.setCurrentIndex(self.CBBackbone.findText(defaultbackbone))

    def setBackBoneOptions(self):
        defaultbackbone = self.parent.dl.Mode.backbone
        self.CBBackbone.clear()
        for bb in self.parent.dl.Mode.getBackbones():
            self.CBBackbone.addItem(bb) 
        self.CBBackbone.setCurrentIndex(self.CBBackbone.findText(defaultbackbone))

    def ArchitectureChanged(self):
        if self.CBArchitecture.currentText() != '':
            self.parent.dl.Mode.setArchitecture(self.CBArchitecture.currentText())
        self.setBackBoneOptions()

    def BackBoneChanged(self):
        if self.CBBackbone.currentText() != '':
            self.parent.dl.Mode.setBackbone(self.CBBackbone.currentText())

    def ModelType(self):
        if self.CBArchitecture.currentText() != '':
            self.parent.dl.Mode.setArchitecture(self.CBArchitecture.currentText())
        if self.CBBackbone.currentText() != '':
            self.parent.dl.Mode.setBackbone(self.CBBackbone.currentText())

    def UsePretrained(self):
        self.parent.dl.Mode.pretrained = self.CBPretrained.isChecked()
        
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


        
        