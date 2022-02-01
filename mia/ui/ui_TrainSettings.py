# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_utils import LabelledAdaptiveDoubleSpinBox, LabelledSpinBox, LabelledDoubleSpinBox, DCDButton, openFolder
from ui.style import styleForm
from dl.optimizer.optimizer import dlOptim
from dl.training.lrschedule import lrMode
from dl.metric.metrics import dlMetric
from dl.loss.losses import dlLoss
from dl.method.mode import dlPreprocess


class Window(object):
    def setupUi(self, Form):

        Form.setWindowTitle('Training Settings') 
        styleForm(Form)

        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(3, 3, 3, 3)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)

        layout = QHBoxLayout()
        self.CBMemory = QCheckBox("Load Full Dataset",self.centralWidget)
        self.CBMemory.setToolTip('Check to load full dataset into memory, uncheck to reload data on each iteration')
        self.CBMemory.setObjectName('nn_TrainInMemory')
        layout.addWidget(self.CBMemory)

        self.SBPredictEvery = LabelledSpinBox ('Predict an image every x epochs', self.centralWidget)
        self.SBPredictEvery.setToolTip('Predict the current selected test image after every x epochs')
        self.SBPredictEvery.SpinBox.setRange(0,100)
        self.SBPredictEvery.SpinBox.setValue(0)
        self.SBPredictEvery.setObjectName('nn_PredictEveryX')
        self.CBPreprocess = QComboBox(self.centralWidget)
        self.CBPreprocess.setObjectName('nn_Preprocessing')
        for pre in dlPreprocess:
            self.CBPreprocess.addItem(pre.name)

        layout.addWidget(self.CBPreprocess)
        self.LPreprocess = QLabel('Preprocessing')
        layout.addWidget(self.LPreprocess)

        self.vlayout.addLayout(layout)
        self.vlayout.addWidget(self.SBPredictEvery)
        
        self.SBBorder = LabelledSpinBox('Ignore border of training images',self.centralWidget)
        self.SBBorder.setObjectName('nn_IgnoreBorder')
        self.SBBorder.setToolTip('Set to remove a border in pixel around each training image')
        self.SBBorder.SpinBox.setRange(0,100)
        self.vlayout.addWidget(self.SBBorder) 
        
        
        self.vlayout.addWidget(self.LossGroup())
        self.vlayout.addWidget(self.TrainValGroup())
        self.vlayout.addWidget(self.LRScheduleGroup())

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

    def LossGroup(self):
        groupBox = QGroupBox("Loss, Metric, Optimizer")
        
        layout = QVBoxLayout()
        hlayout = QHBoxLayout()

        hlayout_2 = QHBoxLayout()
        
        self.LWeights = QLabel("Class Weighting",self.centralWidget)
        self.RBAutoClassWeight = QRadioButton("Auto weighting",self.centralWidget)
        self.RBAutoClassWeight.setToolTip('Check to automatically calculate the weight of each class')
        self.RBAutoClassWeight.setObjectName('nn_AutoClassWeight')
        
        self.RBManuClassWeight = QRadioButton("Manual weighting",self.centralWidget)
        self.RBManuClassWeight.setToolTip('Check to manual set weight of each class')
        self.RBManuClassWeight.setObjectName('nn_ManualClassWeight')
        
        self.RBNoClassWeight = QRadioButton("Disable weighting",self.centralWidget)
        self.RBNoClassWeight.setToolTip('Check to disable class weights')
        self.RBNoClassWeight.setObjectName('nn_NoClassWeight')
        
        hlayout_2.addWidget(self.RBNoClassWeight)
        hlayout_2.addWidget(self.RBAutoClassWeight)
        hlayout_2.addWidget(self.RBManuClassWeight)
        
        hlayout_3 = QHBoxLayout()
        
        self.CBClassWeights = QComboBox(self.centralWidget)
        self.SBWeight = LabelledAdaptiveDoubleSpinBox ('Class weight', self.centralWidget)
        self.SBWeight.SpinBox.setRange(0.0001,1)
        self.SBWeight.setToolTip('Set class weight')
        self.SBWeight.SpinBox.setDecimals(4)
        self.SBWeight.SpinBox.setSingleStep(0.1)
        hlayout_3.addWidget(self.CBClassWeights)
        hlayout_3.addWidget(self.SBWeight)

        layout.addWidget(self.LWeights)
        layout.addLayout(hlayout_2)
        layout.addLayout(hlayout_3)
        
        vlayout_1 = QVBoxLayout()
        self.LLoss = QLabel("Loss",self.centralWidget)
        self.CBLoss = QComboBox(self.centralWidget)
        self.CBLoss.setObjectName('nn_Loss')
        self.CBLoss.setToolTip('Set loss function that is optimized during training')
        vlayout_1.addWidget(self.LLoss)
        vlayout_1.addWidget(self.CBLoss)
        
        vlayout_2 = QVBoxLayout()
        self.LMetric = QLabel("Metric",self.centralWidget)
        self.CBMetrics = QComboBox(self.centralWidget)
        self.CBMetrics.setObjectName('nn_Metric')
        self.CBMetrics.setToolTip('Set metric to measure network performance')
        vlayout_2.addWidget(self.LMetric)
        vlayout_2.addWidget(self.CBMetrics)
        vlayout_3 = QVBoxLayout()
        self.LOptimizer = QLabel("Optimizer",self.centralWidget)
        self.CBOptimizer = QComboBox(self.centralWidget)
        self.CBOptimizer.setObjectName('nn_Optimizer')
        vlayout_3.addWidget(self.LOptimizer)
        vlayout_3.addWidget(self.CBOptimizer)
        
        
        for opt in dlOptim:
            self.CBOptimizer.addItem(opt.name)
        self.CBOptimizer.setToolTip('Set optimizer to minimize loss function')
        hlayout.addLayout(vlayout_1)
        hlayout.addLayout(vlayout_2)
        hlayout.addLayout(vlayout_3)

        layout.addLayout(hlayout)
        
        groupBox.setLayout(layout)

        return groupBox

    def TrainValGroup(self):
        groupBox = QGroupBox("Train/Validation Split")
        
        vlayout = QVBoxLayout()
        h1layout = QHBoxLayout()
        h2layout = QHBoxLayout()
        
        self.RBUseDir = QRadioButton(self.centralWidget)
        self.RBUseDir.setObjectName('ValidationDir')
        self.RBUseDir.setToolTip('Toggle to use a validation directory')
        
        self.BLoadValidationImages = DCDButton(self.centralWidget)
        self.BLoadValidationImages.setToolTip('Select folder with validation images')
        self.BLoadValidationImages.setIcon(QIcon('icons/load.png'))
        
        self.LVal = QLabel(self.centralWidget)
        self.LVal.setText('Validation Folder')
        
        h1layout.addWidget(self.RBUseDir ,stretch=1)
        h1layout.addWidget(self.BLoadValidationImages, stretch=1)
        h1layout.addWidget(self.LVal, stretch=99)
        
        
        self.RBUseSplit = QRadioButton(self.centralWidget)
        self.RBUseSplit.setObjectName('ValidationSplit')
        self.RBUseSplit.setToolTip('Toggle to use a validation split')
        self.STrainVal = QSlider(Qt.Horizontal, self.centralWidget)
        self.STrainVal.setObjectName('nn_TrainValSplit')
        self.STrainVal.setMinimum(20)
        self.STrainVal.setMaximum(100)
        self.STrainVal.setValue(100)
        self.STrainVal.setToolTip('Split training data into a train and validation set.')
        self.LTrainVal = QLabel(self.centralWidget)
        self.LTrainVal.setText('100 % Train 0 % Val')
        

        
        h2layout.addWidget(self.RBUseSplit)
        h2layout.addWidget(self.STrainVal)
        h2layout.addWidget(self.LTrainVal)

        vlayout.addLayout(h1layout)
        vlayout.addLayout(h2layout)
        groupBox.setLayout(vlayout)

        return groupBox

    def LRScheduleGroup(self):
        groupBox = QGroupBox("Learning Rate Schedule")
        
        layout = QVBoxLayout()
        hlayout = QHBoxLayout()

        self.RBConstant = QRadioButton(self.centralWidget)
        self.RBConstant.setObjectName('nn_ConstantLR')
        self.RBConstant.setText("Constant")
        self.RBConstant.setToolTip('Toggle to set constant learning rate')


        self.RBRlroP = QRadioButton(self.centralWidget)
        self.RBRlroP.setObjectName('nn_ReduceLR')
        self.RBRlroP.setText("on Plateau")
        self.RBRlroP.setToolTip('Toggle to reduce learning rate on plateau')

        self.RBLRschedule = QRadioButton(self.centralWidget)
        self.RBLRschedule.setObjectName('nn_ScheduleLR')
        self.RBLRschedule.setText("Schedule")
        self.RBLRschedule.setToolTip('Toggle to set learning rate schedule')

        hlayout.addWidget(self.RBConstant)
        hlayout.addWidget(self.RBRlroP)
        hlayout.addWidget(self.RBLRschedule)
        layout.addLayout(hlayout)

        self.SBlr = LabelledAdaptiveDoubleSpinBox ('Learning Rate', self.centralWidget)
        self.SBlr.setObjectName('nn_Learning_Rate')
        self.SBlr.SpinBox.setRange(0.000001,1)
        self.SBlr.setToolTip('Set learning rate')
        self.SBlr.SpinBox.setDecimals(6)
        self.SBlr.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBlr)
        
        self.SBReduction = LabelledAdaptiveDoubleSpinBox ('Reduction Factor', self.centralWidget)
        self.SBReduction.setObjectName('nn_LRReduction')
        self.SBReduction.SpinBox.setRange(0.001,1)
        self.SBReduction.SpinBox.setDecimals(3)
        self.SBReduction.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBReduction)

        self.SBEveryx = LabelledSpinBox ('Reduce every x epochs', self.centralWidget)
        self.SBEveryx.setObjectName('nn_LREpochs')
        self.SBEveryx.SpinBox.setRange(1,1000)
        self.SBEveryx.SpinBox.setValue(25)
        layout.addWidget(self.SBEveryx)
        layout.addStretch()

        groupBox.setLayout(layout)

        return groupBox


class TrainSettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TrainSettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBMemory.setChecked(self.parent.parent.dl.TrainInMemory)
        
        self.CBOptimizer.setCurrentIndex(self.CBOptimizer.findText(self.parent.parent.dl.optimizer.Optimizer.name))
        self.SBlr.SpinBox.setValue(self.parent.parent.dl.learning_rate)
        self.SBlr.SpinBox.setMinimum(self.parent.parent.dl.lrschedule.minlr)
        self.SBPredictEvery.SpinBox.setValue(self.parent.parent.predictionRate)
        
        self.CBMemory.stateChanged.connect(self.InMemoryChanged)
        
        self.RBNoClassWeight.setChecked(not self.parent.parent.dl.data.autocalcClassWeights)
        self.RBNoClassWeight.toggled.connect(self.changeClassWeightSettings)
        self.RBAutoClassWeight.toggled.connect(self.changeClassWeightSettings)
        self.RBManuClassWeight.toggled.connect(self.changeClassWeightSettings)
        self.CBClassWeights.currentIndexChanged.connect(self.changeSelectedClass)
        self.SBWeight.SpinBox.valueChanged.connect(self.changeClassWeights)

        self.CBOptimizer.currentIndexChanged.connect(self.changeOptimizer)
        self.CBMetrics.currentIndexChanged.connect(self.changeMetric)
        self.CBLoss.currentIndexChanged.connect(self.changeLoss)
        self.CBPreprocess.setCurrentIndex(self.CBOptimizer.findText(self.parent.parent.dl.preprocess.name))
        self.CBPreprocess.currentIndexChanged.connect(self.changePreprocessing)
        
        self.SBBorder.SpinBox.setValue(self.parent.parent.dl.data.IgnoreBorder)
        self.SBBorder.SpinBox.valueChanged.connect(self.changeBorder)

        self.RBConstant.toggled.connect(self.LROption)
        self.RBRlroP.toggled.connect(self.LROption)
        self.RBLRschedule.toggled.connect(self.LROption)
        
        self.RBUseDir.toggled.connect(self.valsetChanged)
        self.RBUseSplit.toggled.connect(self.valsetChanged)
        self.BLoadValidationImages.clicked.connect(self.selectValDir)
        self.RBUseDir.setChecked(self.parent.parent.dl.data.UseValidationDir)
        
        
        self.SBlr.SpinBox.valueChanged.connect(self.changeLR)
        self.SBReduction.SpinBox.valueChanged.connect(self.changeReductionFactor)
        self.SBEveryx.SpinBox.valueChanged.connect(self.changeIntervall)
        self.SBPredictEvery.SpinBox.valueChanged.connect(self.predictionRateChanged)
        
        self.STrainVal.valueChanged.connect(self.changeTrainTestSplit)
        self.STrainVal.setValue(self.parent.parent.dl.data.TrainTestSplit*100)
        
        self.RBRlroP.setChecked(True)
        self.LROption()
        self.updateLossesAndMetrics()
        self.changeClassWeightSettings()

    def InMemoryChanged(self):
        self.parent.parent.dl.TrainInMemory = self.CBMemory.isChecked()
        
    def predictionRateChanged(self):
        self.parent.parent.predictionRate = self.SBPredictEvery.SpinBox.value()
        
    def changeOptimizer(self, i):
        self.parent.parent.dl.optimizer.setOptimizer(dlOptim[self.CBOptimizer.currentText()])
        self.parent.SBLearningRate.SpinBox.setValue(self.parent.parent.dl.learning_rate)
        
    def LROption(self):
        if self.RBConstant.isChecked():
            self.SBReduction.hide()
            self.SBEveryx.hide()
            self.parent.parent.dl.lrschedule.setMode(lrMode.Constant)
        if self.RBRlroP.isChecked():
            self.SBReduction.show()
            self.SBEveryx.show()
            self.SBEveryx.Label.setText('Plateau for x epochs')   
            self.parent.parent.dl.lrschedule.setMode(lrMode.ReduceOnPlateau)
        if self.RBLRschedule.isChecked():
            self.SBReduction.show()
            self.SBEveryx.show()
            self.SBEveryx.Label.setText('Reduce every x epochs')
            self.parent.parent.dl.lrschedule.setMode(lrMode.Schedule)
        self.SBEveryx.SpinBox.setValue(self.parent.parent.dl.lrschedule.ReduceEveryXEpoch)
        self.SBReduction.SpinBox.setValue(self.parent.parent.dl.lrschedule.ReductionFactor)
            
    def changeReductionFactor(self):
        self.parent.parent.dl.lrschedule.ReductionFactor = self.SBReduction.SpinBox.value() 
        
    def changeTrainTestSplit(self):
        self.parent.parent.dl.data.TrainTestSplit = self.STrainVal.value() / 100.
        text = '%d %% Train  %d %% Val' % (self.parent.parent.dl.data.TrainTestSplit*100, 100-self.parent.parent.dl.data.TrainTestSplit*100)
        self.LTrainVal.setText(text)
        
    def changeIntervall(self):
        self.parent.parent.dl.lrschedule.ReduceEveryXEpoch = self.SBEveryx.SpinBox.value()
 
    def changeLR(self):
        self.parent.SBLearningRate.SpinBox.setValue(self.SBlr.SpinBox.value())
            
    def silentlyUpdateLR(self, value):
        self.SBlr.SpinBox.blockSignals(True)
        self.SBlr.SpinBox.setValue(value)
        self.SBlr.SpinBox.blockSignals(False)

    def changeLoss(self):
        if self.CBLoss.currentText() != '': 
            self.parent.parent.dl.Mode.loss.loss = dlLoss[self.CBLoss.currentText()]

    def changePreprocessing(self):
        if self.CBPreprocess.currentText() != '': 
            self.parent.parent.dl.preprocess = dlPreprocess[self.CBPreprocess.currentText()]

    def changeMetric(self):
        if self.CBMetrics.currentText() != '':
            self.parent.parent.dl.Mode.metric.metric = dlMetric[self.CBMetrics.currentText()]

    def updateLossesAndMetrics(self):
        self.CBLoss.clear()
        self.CBMetrics.clear()
        default_loss = self.parent.parent.dl.Mode.loss.loss.name
        default_metric = self.parent.parent.dl.Mode.metric.metric.name
        for loss in self.parent.parent.dl.Mode.loss.supportedlosses:
            self.CBLoss.addItem(loss.name)
        for metric in self.parent.parent.dl.Mode.metric.supportedmetrics:
            self.CBMetrics.addItem(metric.name)
        self.CBLoss.setCurrentIndex(self.CBLoss.findText(default_loss))
        self.CBMetrics.setCurrentIndex(self.CBMetrics.findText(default_metric))

    def changeClassWeightSettings(self):
        self.parent.parent.dl.data.autocalcClassWeights = self.RBAutoClassWeight.isChecked()
        self.CBClassWeights.clear()
        
        if self.parent.parent.NumOfClasses() > 2:
            for c in range(self.parent.parent.NumOfClasses()):
                self.CBClassWeights.addItem(self.parent.parent.classList.getClassName(c))
        else:
            self.CBClassWeights.addItem(self.parent.parent.classList.getClassName(0))
            self.CBClassWeights.addItem(self.parent.parent.classList.getClassName(1))
        self.CBClassWeights.setCurrentIndex(1) 
        if self.RBManuClassWeight.isChecked():
            self.CBClassWeights.setEnabled(True)
            self.SBWeight.setEnabled(True)
            self.changeClassWeights()
        else:
            self.CBClassWeights.setEnabled(False)
            self.SBWeight.setEnabled(False)
            
        if self.RBNoClassWeight.isChecked():
            self.parent.parent.dl.data.resetClassWeights()
            
    def changeSelectedClass(self):
        if self.CBClassWeights.currentIndex() >= 0:
            self.SBWeight.SpinBox.setValue(self.parent.parent.classList.getClassWeight(self.CBClassWeights.currentIndex()))
            
    def changeClassWeights(self):
        self.parent.parent.classList.setClassWeight(self.CBClassWeights.currentIndex(),self.SBWeight.SpinBox.value())
        if self.parent.parent.NumOfClasses() == 2:
            idx = 1 - self.CBClassWeights.currentIndex()
            self.parent.parent.classList.setClassWeight(idx,1-self.SBWeight.SpinBox.value())
        self.parent.parent.setClassWeights()

            
    def valsetChanged(self):
        self.parent.parent.dl.data.UseValidationDir = self.RBUseDir.isChecked()
        self.STrainVal.setEnabled(self.RBUseSplit.isChecked())
        self.BLoadValidationImages.setEnabled(self.RBUseDir.isChecked())
        
        

    def selectValDir(self):
        folder = openFolder("Select Validation Image Folder") 
        if folder:
            self.parent.parent.files.validationpath = folder
            self.LVal.setText(folder)
        
    def changeBorder(self):
        self.parent.parent.dl.data.IgnoreBorder = self.SBBorder.SpinBox.value()
