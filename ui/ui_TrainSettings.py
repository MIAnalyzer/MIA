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
from dl.optimizer.optimizer import dlOptim
from dl.training.lrschedule import lrMode
from dl.metric.metrics import dlMetric
from dl.loss.losses import dlLoss
from dl.data.imagedata import dlPreprocess


class Window(object):
    def setupUi(self, Form):
        width = 400
        height= 420
        Form.setWindowTitle('Training Settings') 
        styleForm(Form)
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 3, 3, 3)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)

        layout = QHBoxLayout(self.centralWidget)
        self.CBMemory = QCheckBox("Load Full Dataset",self.centralWidget)
        self.CBMemory.setToolTip('Check to load full dataset into memory, uncheck to reload data on each iteration')
        self.CBMemory.setObjectName('TrainInMemory')
        layout.addWidget(self.CBMemory)

        self.SBPredictEvery = LabelledSpinBox ('Predict an image every x epochs', self.centralWidget)
        self.SBPredictEvery.setToolTip('Predict the current selected test image after every x epochs')
        self.SBPredictEvery.SpinBox.setRange(0,100)
        self.SBPredictEvery.SpinBox.setValue(0)
        self.SBPredictEvery.setObjectName('PredictEveryX')
        self.CBPreprocess = QComboBox(self.centralWidget)
        self.CBPreprocess.setObjectName('Preprocessing')
        for pre in dlPreprocess:
            self.CBPreprocess.addItem(pre.name)

        layout.addWidget(self.CBPreprocess)
        self.LPreprocess = QLabel('Preprocessing')
        layout.addWidget(self.LPreprocess)

        self.vlayout.addLayout(layout)
        self.vlayout.addWidget(self.SBPredictEvery)
        self.vlayout.addWidget(self.LossGroup())
        self.vlayout.addWidget(self.TrainValGroup())
        self.vlayout.addWidget(self.LRScheduleGroup())

    def LossGroup(self):
        groupBox = QGroupBox("Loss, Metric, Optimizer")
        
        layout = QVBoxLayout(self.centralWidget)
        hlayout = QHBoxLayout(self.centralWidget)

        self.CBClassWeight = QCheckBox("Calculate class weighting",self.centralWidget)
        self.CBClassWeight.setToolTip('Check to automatically calculate the weight of each class')
        self.CBClassWeight.setObjectName('CallculateClassWeight')
        layout.addWidget(self.CBClassWeight)
        self.CBLoss = QComboBox(self.centralWidget)
        self.CBLoss.setObjectName('Loss')
        # self.CBLoss.addItem("Focal Loss")
        self.CBLoss.setToolTip('Set loss function that is optimized during training')
        self.CBMetrics = QComboBox(self.centralWidget)
        # self.CBMetrics.addItem("IoU")
        self.CBMetrics.setObjectName('Metric')
        self.CBMetrics.setToolTip('Set metric to measure network performance')
        self.CBOptimizer = QComboBox(self.centralWidget)
        self.CBOptimizer.setObjectName('Optimizer')
        for opt in dlOptim:
            self.CBOptimizer.addItem(opt.name)
        self.CBOptimizer.setToolTip('Set optimizer to minimize loss function')
        hlayout.addWidget(self.CBLoss)
        hlayout.addWidget(self.CBMetrics)
        hlayout.addWidget(self.CBOptimizer)

        layout.addLayout(hlayout)
        
        groupBox.setLayout(layout)

        return groupBox

    def TrainValGroup(self):
        groupBox = QGroupBox("Train/Validation Split")
        
        hlayout = QHBoxLayout(self.centralWidget)
        self.STrainVal = QSlider(Qt.Horizontal, self.centralWidget)
        self.STrainVal.setObjectName('TrainValSplit')
        self.STrainVal.setMinimum(20)
        self.STrainVal.setMaximum(100)
        self.STrainVal.setValue(100)
        self.STrainVal.setToolTip('Split training data into a train and validation set.')
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
        self.RBConstant.setObjectName('ConstantLR')
        self.RBConstant.setText("Constant")
        self.RBConstant.setToolTip('Toggle to set constant learning rate')


        self.RBRlroP = QRadioButton(self.centralWidget)
        self.RBRlroP.setObjectName('ReduceLR')
        self.RBRlroP.setText("on Plateau")
        self.RBRlroP.setToolTip('Toggle to reduce learning rate on plateau')

        self.RBLRschedule = QRadioButton(self.centralWidget)
        self.RBLRschedule.setObjectName('ScheduleLR')
        self.RBLRschedule.setText("Schedule")
        self.RBLRschedule.setToolTip('Toggle to set learning rate schedule')

        hlayout.addWidget(self.RBConstant)
        hlayout.addWidget(self.RBRlroP)
        hlayout.addWidget(self.RBLRschedule)
        layout.addLayout(hlayout)

        self.SBlr = LabelledAdaptiveDoubleSpinBox ('Learning Rate', self.centralWidget)
        self.SBlr.setObjectName('Learning_Rate')
        self.SBlr.SpinBox.setRange(0.000001,1)
        self.SBlr.setToolTip('Set learning rate')
        self.SBlr.SpinBox.setDecimals(6)
        self.SBlr.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBlr)
        
        self.SBReduction = LabelledAdaptiveDoubleSpinBox ('Reduction Factor', self.centralWidget)
        self.SBReduction.setObjectName('LRReduction')
        self.SBReduction.SpinBox.setRange(0.001,1)
        self.SBReduction.SpinBox.setDecimals(3)
        self.SBReduction.SpinBox.setSingleStep(0.1)
        layout.addWidget(self.SBReduction)

        self.SBEveryx = LabelledSpinBox ('Reduce every x epochs', self.centralWidget)
        self.SBEveryx.setObjectName('LREpochs')
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
        
        self.CBClassWeight.setChecked(self.parent.parent.dl.data.autocalcClassWeights)
        self.CBClassWeight.stateChanged.connect(self.calculateClassWeight)
        self.CBOptimizer.currentIndexChanged.connect(self.changeOptimizer)
        self.CBMetrics.currentIndexChanged.connect(self.changeMetric)
        self.CBLoss.currentIndexChanged.connect(self.changeLoss)
        self.CBPreprocess.setCurrentIndex(self.CBOptimizer.findText(self.parent.parent.dl.preprocess.name))
        self.CBPreprocess.currentIndexChanged.connect(self.changePreprocessing)

        self.RBConstant.toggled.connect(self.LROption)
        self.RBRlroP.toggled.connect(self.LROption)
        self.RBLRschedule.toggled.connect(self.LROption)
        
        self.SBlr.SpinBox.valueChanged.connect(self.changeLR)
        self.SBReduction.SpinBox.valueChanged.connect(self.changeReductionFactor)
        self.SBEveryx.SpinBox.valueChanged.connect(self.changeIntervall)
        self.SBPredictEvery.SpinBox.valueChanged.connect(self.predictionRateChanged)
        
        self.STrainVal.valueChanged.connect(self.changeTrainTestSplit)
        self.STrainVal.setValue(self.parent.parent.dl.data.TrainTestSplit*100)
        
        self.RBConstant.setChecked(True)
        self.LROption()
        self.updateLossesAndMetrics()

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

    def calculateClassWeight(self):
        self.parent.parent.dl.data.autocalcClassWeights = self.CBClassWeight.isChecked()
        self.parent.parent.checkForTrainingWarnings()


