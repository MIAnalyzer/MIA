# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:02 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import multiprocessing
import tensorflow as tf

from ui.ui_utils import LabelledSpinBox, LabelledDoubleSpinBox
from ui.style import styleForm

class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Settings') 
        styleForm(Form)

        self.centralWidget = QWidget(Form)
        
        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(3, 10, 3, 3)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)
        
        self.dlParams = self.DLGroup()
        self.vlayout.addWidget(self.dlParams)
        
        self.dispParams = self.DispGroup()
        self.vlayout.addWidget(self.dispParams)

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

        
    def DLGroup(self):
        groupBox = QGroupBox("Deep Learning Settings")
        vlayout = QVBoxLayout()
        
        vlayout.setContentsMargins(1, 1, 1, 1)
        vlayout.setSpacing(1)
        
        hlayout = QHBoxLayout()
        self.CBgpu = QComboBox(self.centralWidget)
        self.CBgpu.setObjectName('GPUSettings')

        devices = tf.config.experimental.list_physical_devices()
        self.CBgpu.addItem("default")
        for dev in devices:
            self.CBgpu.addItem(dev.name)                  
        self.CBgpu.addItem("all gpus")
        
        self.Lgpu = QLabel(self.centralWidget)
        self.Lgpu.setText('GPU settings')
        hlayout.addWidget(self.CBgpu)
        hlayout.addWidget(self.Lgpu)
        vlayout.addLayout(hlayout)     
        
        self.CBMixedPrecision = QCheckBox("Mixed Precision Computation",self.centralWidget)
        self.CBMixedPrecision.setObjectName('Mixed_precision')
        self.CBMixedPrecision.setToolTip('Select to use mixed precision during training, reduces gpu memory and increases computation speed on modern gpus')
        vlayout.addWidget(self.CBMixedPrecision)
        
        self.CBSeparateLabels = QCheckBox("Separate Labels of Stacks",self.centralWidget)
        self.CBSeparateLabels.setObjectName('Separate_Labels')
        self.CBSeparateLabels.setToolTip('Select to use different labels for each frame in an image stack')
        self.CBSeparateLabels.setEnabled(True)
        vlayout.addWidget(self.CBSeparateLabels)
                
        
        self.CBTTA = QCheckBox("Test Time Augmentation",self.centralWidget)
        self.CBTTA.setObjectName('TestTimeAugmentation')
        self.CBTTA.setToolTip('Select to use test time augmentation')
        vlayout.addWidget(self.CBTTA)
        
        self.SBBorderOverlap = LabelledSpinBox('Model Overlap',self.centralWidget)
        self.SBBorderOverlap.setObjectName('ModelOverlap')
        self.SBBorderOverlap.SpinBox.setRange(0,1024)
        self.SBBorderOverlap.setToolTip('Set tile overlap during prediction')
        vlayout.addWidget(self.SBBorderOverlap)
        

        self.CBWeightsonly = QCheckBox("Load/Save only model weights",self.centralWidget)
        self.CBWeightsonly.setObjectName('Weightsonly')
        self.CBWeightsonly.setToolTip('Select to save/load only model weights instead of complete model setup')
        vlayout.addWidget(self.CBWeightsonly)

        self.SBThreads = LabelledSpinBox('Worker threads',self.centralWidget)
        self.SBThreads.setObjectName('NumThreads')
        self.SBThreads.setToolTip('Set number of worker threads')
        self.SBThreads.SpinBox.setRange(1,multiprocessing.cpu_count())
        vlayout.addWidget(self.SBThreads)  
        
        self.SBScaleFactor = LabelledDoubleSpinBox('Image Down Scaling',self.centralWidget)
        # do not give a name as scaling is named in Training
        self.SBScaleFactor.SpinBox.setRange(0.1,1)
        self.SBScaleFactor.setToolTip('Set image reduction factor')
        self.SBScaleFactor.SpinBox.setSingleStep(0.1)
        self.SBScaleFactor.SpinBox.setDecimals(1)
        vlayout.addWidget(self.SBScaleFactor)
        

        
        
        groupBox.setLayout(vlayout)
        return groupBox
    
    def DispGroup(self):
        groupBox = QGroupBox("Display Settings")
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(1, 1, 1, 1)
        vlayout.setSpacing(1)        
        
                
        self.CBShapeNumbers = QCheckBox("Show Object Numbers",self.centralWidget)
        self.CBShapeNumbers.setToolTip('Select to show number below each shape')
        self.CBShapeNumbers.setObjectName('ShowShapeNumbers')
        vlayout.addWidget(self.CBShapeNumbers)
        
        self.CBFastDrawing = QCheckBox("Fast Drawing",self.centralWidget)
        self.CBFastDrawing.setToolTip('Reduces performance needs during drawing. Might be used for large images.')
        self.CBFastDrawing.setObjectName('FastDrawing')
        vlayout.addWidget(self.CBFastDrawing)
        

        self.SBFontSize = LabelledSpinBox('Font Size',self.centralWidget)
        self.SBFontSize.setToolTip('Set font size')
        self.SBFontSize.setObjectName('FontSize')
        self.SBFontSize.SpinBox.setRange(4,32)
        vlayout.addWidget(self.SBFontSize)

        self.SBPenSize = LabelledSpinBox('Pen Size',self.centralWidget)
        self.SBPenSize.SpinBox.setRange(1,15)
        self.SBPenSize.setObjectName('PenSize')
        self.SBPenSize.setToolTip('Set pen size')
        vlayout.addWidget(self.SBPenSize)
        
        self.h2layout = QHBoxLayout()
        self.STransparency = QSlider(Qt.Horizontal, self.centralWidget)
        self.STransparency.setToolTip('Set opacity of contours')
        self.STransparency.setObjectName('ContourTransparency')
        self.STransparency.setMinimum(0)
        self.STransparency.setMaximum(255)
        
        self.LTransparency = QLabel(self.centralWidget)
        self.LTransparency.setText(' Contour Opacity')
        self.h2layout.addWidget(self.STransparency, stretch = 1)
        self.h2layout.addWidget(self.LTransparency, stretch = 1)
        vlayout.addLayout(self.h2layout)
        groupBox.setLayout(vlayout)
        return groupBox
        

class SettingsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(SettingsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        self.CBShapeNumbers.setChecked(self.parent.canvas.drawShapeNumber)
        self.CBFastDrawing.setChecked(self.parent.canvas.fastPainting)
        self.CBSeparateLabels.setChecked(self.parent.separateStackLabels)
        self.SBPenSize.SpinBox.setValue(self.parent.canvas.pen_size)
        self.SBFontSize.SpinBox.setValue(self.parent.canvas.FontSize)
        self.STransparency.setValue(self.parent.canvas.ContourTransparency)
        self.SBThreads.SpinBox.setValue(self.parent.maxworker)
        self.SBScaleFactor.SpinBox.setValue(self.parent.dl.ImageScaleFactor)
        self.CBTTA.setChecked(self.parent.dl.tta)
        self.SBBorderOverlap.SpinBox.setValue(self.parent.dl.borderremoval)
        self.CBWeightsonly.setChecked(self.parent.saveload_modelweights_only)
        self.CBMixedPrecision.setChecked(self.parent.dl.mixed_precision)
        
        
        self.CBShapeNumbers.stateChanged.connect(self.showContourNumbers)
        self.CBFastDrawing.stateChanged.connect(self.fastDrawing)
        self.CBSeparateLabels.stateChanged.connect(self.separateLabels)
        self.SBPenSize.SpinBox.valueChanged.connect(self.setPenSize)
        self.SBFontSize.SpinBox.valueChanged.connect(self.setFontSize)
        self.STransparency.valueChanged.connect(self.setTransparency)
        self.CBgpu.currentIndexChanged.connect(self.setGPU)
        self.SBThreads.SpinBox.valueChanged.connect(self.setNumThreads)
        self.SBScaleFactor.SpinBox.valueChanged.connect(self.ScaleFactorChanged)
        self.CBTTA.stateChanged.connect(self.ttaChanged)
        self.SBBorderOverlap.SpinBox.valueChanged.connect(self.setBorder)
        self.CBWeightsonly.stateChanged.connect(self.weightsonlyChanged)
        self.CBMixedPrecision.stateChanged.connect(self.precisionChanged)
        
        self.CBgpu.setCurrentIndex(0)
        
    def setGPU(self):
        # finds cpu and gpu
        devices = tf.config.experimental.list_physical_devices()
        gpus = tf.config.experimental.list_physical_devices('GPU') 
        dev = self.CBgpu.currentIndex() - 1 
        if devices:
            try:
                if dev == -1:
                    # tensorflow decides 
                    tf.config.experimental.set_visible_devices(gpus, 'GPU')
                elif dev >= len(devices):
                    # currently unsupported
                    self.parent.PopupWarning('Use of multiple GPUs currently not supported')
                elif devices[dev].device_type == 'CPU':
                    # set cpu
                    tf.config.experimental.set_visible_devices([], 'GPU')
                else:
                    # set gpu
                    tf.config.experimental.set_visible_devices(devices[dev], 'GPU')

            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                self.parent.PopupWarning('GPU settings only take effect upon program start')
    
    def fastDrawing(self):
        self.parent.canvas.fastPainting = self.CBFastDrawing.isChecked()
        
       
    def separateLabels(self):
        self.parent.separateStackLabels = self.CBSeparateLabels.isChecked()
        self.parent.training_form.changeStackLabelOption()
        self.parent.enableCopyStackLabels()
        self.parent.canvas.ReloadLabels()
              
    def ttaChanged(self):
        self.parent.dl.tta = self.CBTTA.isChecked()
        
    def setBorder(self):
        self.parent.dl.borderremoval = self.SBBorderOverlap.SpinBox.value()
        
    def weightsonlyChanged(self):
        self.parent.saveload_modelweights_only = self.CBWeightsonly.isChecked()
        
    def precisionChanged(self):
        self.parent.dl.mixed_precision = self.CBMixedPrecision.isChecked()
        
    def ScaleFactorChanged(self):
        self.parent.training_form.SBScaleFactor.SpinBox.setValue(self.SBScaleFactor.SpinBox.value())
        
    def silentlyUpdateScale(self, value):
        self.SBScaleFactor.SpinBox.blockSignals(True)
        self.SBScaleFactor.SpinBox.setValue(value)
        self.SBScaleFactor.SpinBox.blockSignals(False)
        
    def setNumThreads(self):
        self.parent.setWorkers(self.SBThreads.SpinBox.value())
    
    def showContourNumbers(self):
        self.parent.canvas.drawShapeNumber = self.CBShapeNumbers.isChecked()
        self.parent.canvas.redrawImage()
        
    def setPenSize(self):
        self.parent.canvas.pen_size = self.SBPenSize.SpinBox.value()
        self.parent.canvas.redrawImage()
        
    def setFontSize(self):
        self.parent.canvas.FontSize = self.SBFontSize.SpinBox.value()
        self.parent.canvas.redrawImage()
        
    def setTransparency(self):
        self.parent.canvas.ContourTransparency = self.STransparency.value()
        self.parent.canvas.redrawImage()
        
    def closeEvent(self, event):
        self.parent.canvas.redrawImage()