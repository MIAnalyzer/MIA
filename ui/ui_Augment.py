# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_utils import LabelledSpinBox, LabelledDoubleSpinBox
from ui.style import styleForm



class Window(object):
    def setupUi(self, Form):
        width = 400
        height= 500
        Form.setWindowTitle('Augmentation') 
        styleForm(Form)
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(3, 3, 3, 3)
        self.vlayout.setSpacing(0)
        self.centralWidget.setLayout(self.vlayout)

        self.augmentations = self.AugmentationGroup()
        self.augmentations.setCheckable(True)
        self.vlayout.addWidget(self.augmentations)


    def AugmentationGroup(self):
       
        groupBox = QGroupBox("Augmentations")

        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        sizelayout = QHBoxLayout(self.centralWidget)
        sizelayout.setSpacing(0)
        sizelayout.setContentsMargins(1, 1, 1, 1)
        self.SBModelInputSize_x = LabelledSpinBox(' Input width', self.centralWidget)
        self.SBModelInputSize_x.setToolTip('Set model input width')
        self.SBModelInputSize_x.SpinBox.setRange(1,1024)
        self.SBModelInputSize_y = LabelledSpinBox(' Input height', self.centralWidget)
        self.SBModelInputSize_y.setToolTip('Set model input height')
        self.SBModelInputSize_y.SpinBox.setRange(1,1024)

        sizelayout.addWidget(self.SBModelInputSize_x)
        sizelayout.addWidget(self.SBModelInputSize_y)
        layout.addItem(sizelayout)

        hlayout = QHBoxLayout(self.centralWidget)
        self.Flip = self.FlipGroup()
        hlayout.addWidget(self.Flip)
        self.Noise = self.NoiseGroup()
        hlayout.addWidget(self.Noise)
        layout.addItem(hlayout)

        self.Affine = self.AffineGroup()
        self.Affine.setCheckable(True)
        layout.addWidget(self.Affine)

        groupBox.setLayout(layout)

        return groupBox


    def FlipGroup(self):
        groupBox = QGroupBox("Flip")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.CBFliplr = QCheckBox("Flip horizontal",self.centralWidget)
        self.CBFliplr.setToolTip('Check to flip images horizontally')
        self.CBFliptb = QCheckBox("Flip vertical",self.centralWidget)
        self.CBFliptb.setToolTip('Check to flip images vertically')

        layout.addWidget(self.CBFliplr)
        layout.addWidget(self.CBFliptb)


        groupBox.setLayout(layout)
        return groupBox


    def NoiseGroup(self):
        groupBox = QGroupBox("Noise")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBNoise = LabelledSpinBox(' Blur (%)', self.centralWidget)
        self.SBNoise.SpinBox.setRange(0,100)
        self.SBNoise.setToolTip('Percentage to apply gaussian blur to the input image')

        self.SBPieceWise = LabelledSpinBox(' Piecewise (%)', self.centralWidget)
        self.SBPieceWise.SpinBox.setRange(0,100)
        self.SBPieceWise.setToolTip('Percentage to apply piecewise affine transformation')

        self.SBDropout = LabelledSpinBox(' Dropout (%)', self.centralWidget)
        self.SBDropout.SpinBox.setRange(0,100)
        self.SBDropout.setToolTip('Percentage to apply piecewise dropout')

        layout.addWidget(self.SBNoise)
        layout.addWidget(self.SBPieceWise)
        layout.addWidget(self.SBDropout)

        groupBox.setLayout(layout)
        return groupBox

    def AffineGroup(self):

        groupBox = QGroupBox("Affine")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        hlayout = QHBoxLayout(self.centralWidget)
        self.SBPercentage = LabelledSpinBox(' Apply Affine (%)', self.centralWidget)
        self.SBPercentage.setToolTip('Percentage to apply affine transformation')
        self.SBPercentage.SpinBox.setRange(0,100)
        hlayout.addWidget(self.SBPercentage)

        layout.addLayout(hlayout)

        grid = QGridLayout(self.centralWidget)
        grid.addWidget(self.ScaleGroup(), 0,0)
        grid.addWidget(self.TranslateGroup(), 0,1)
        grid.addWidget(self.ShearGroup(), 1,0)
        grid.addWidget(self.RotateGroup(), 1,1)
        layout.addItem(grid)
        groupBox.setLayout(layout)
        return groupBox

    def ScaleGroup(self):
        groupBox = QGroupBox("Scale")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBScaleMin = LabelledDoubleSpinBox(' Scale Min (factor)', self.centralWidget)
        self.SBScaleMin.setToolTip('Set scale maximum percentage')
        self.SBScaleMin.SpinBox.setSingleStep(0.1)
        self.SBScaleMin.SpinBox.setDecimals(1)
        self.SBScaleMin.SpinBox.setRange(0,1)

        self.SBScaleMax = LabelledDoubleSpinBox(' Scale Max (factor)', self.centralWidget)
        self.SBScaleMax.setToolTip('Set scale minimum percentage')
        self.SBScaleMax.SpinBox.setSingleStep(0.1)
        self.SBScaleMax.SpinBox.setDecimals(1)
        self.SBScaleMax.SpinBox.setRange(1,10)

        layout.addWidget(self.SBScaleMin)
        layout.addWidget(self.SBScaleMax)


        groupBox.setLayout(layout)
        return groupBox

    def TranslateGroup(self):
        groupBox = QGroupBox("Translate")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBTranslateMin = LabelledDoubleSpinBox(' Translate Min (%)', self.centralWidget)
        self.SBTranslateMin.setToolTip('Set translation minimun percent')
        self.SBTranslateMin.SpinBox.setSingleStep(0.1)
        self.SBTranslateMin.SpinBox.setDecimals(1)
        self.SBTranslateMin.SpinBox.setRange(-1,0)

        self.SBTranslateMax = LabelledDoubleSpinBox(' Translate Max (%)', self.centralWidget)
        self.SBTranslateMax.setToolTip('Set translation minimun percent')
        self.SBTranslateMax.SpinBox.setSingleStep(0.1)
        self.SBTranslateMax.SpinBox.setDecimals(1)
        self.SBTranslateMax.SpinBox.setRange(0,1)

        layout.addWidget(self.SBTranslateMin)
        layout.addWidget(self.SBTranslateMax)

        groupBox.setLayout(layout)
        return groupBox

    def ShearGroup(self):
        groupBox = QGroupBox("Shear")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBShearMin = LabelledSpinBox(' Shear Min (deg)', self.centralWidget)
        self.SBShearMin.setToolTip('Set shear minimum')
        self.SBShearMin.SpinBox.setRange(-45,0)

        self.SBShearMax = LabelledSpinBox(' Shear Max (deg)', self.centralWidget)
        self.SBShearMax.setToolTip('Set shear maximum')
        self.SBShearMax.SpinBox.setRange(0,45)

        layout.addWidget(self.SBShearMin)
        layout.addWidget(self.SBShearMax)

        groupBox.setLayout(layout)
        return groupBox

    def RotateGroup(self):
        groupBox = QGroupBox("Rotate")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBRotMin = LabelledSpinBox(' Rotation Min (deg)', self.centralWidget)
        self.SBRotMin.setToolTip('Set rotation range minimum')
        self.SBRotMin.SpinBox.setRange(-45,0)

        self.SBRotMax = LabelledSpinBox(' Rotation Max (deg)', self.centralWidget)
        self.SBRotMax.setToolTip('Set rotation range maximum')
        self.SBRotMax.SpinBox.setRange(0,45)

        layout.addWidget(self.SBRotMin)
        layout.addWidget(self.SBRotMax)

        groupBox.setLayout(layout)
        return groupBox





class AugmentWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(AugmentWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        
        self.SBModelInputSize_x.SpinBox.setValue(self.parent.dl.augmentation.outputwidth)
        self.SBModelInputSize_y.SpinBox.setValue(self.parent.dl.augmentation.outputheight)
        
        self.CBFliplr.setChecked(self.parent.dl.augmentation.flip_horz)
        self.CBFliptb.setChecked(self.parent.dl.augmentation.flip_vert)
        
        self.SBNoise.SpinBox.setValue(self.parent.dl.augmentation.prob_gaussian *100)
        self.SBPieceWise.SpinBox.setValue(self.parent.dl.augmentation.prob_piecewise *100)
        self.SBDropout.SpinBox.setValue(self.parent.dl.augmentation.prob_dropout *100)
        
        self.SBPercentage.SpinBox.setValue(self.parent.dl.augmentation.prob_affine *100)
        self.SBScaleMin.SpinBox.setValue(self.parent.dl.augmentation.scale_min)
        self.SBScaleMax.SpinBox.setValue(self.parent.dl.augmentation.scale_max)
        self.SBTranslateMin.SpinBox.setValue(self.parent.dl.augmentation.translate_min)
        self.SBTranslateMax.SpinBox.setValue(self.parent.dl.augmentation.translate_max)
        self.SBShearMin.SpinBox.setValue(self.parent.dl.augmentation.shear_min)
        self.SBShearMax.SpinBox.setValue(self.parent.dl.augmentation.shear_max)
        self.SBRotMin.SpinBox.setValue(self.parent.dl.augmentation.rotate_min)
        self.SBRotMax.SpinBox.setValue(self.parent.dl.augmentation.rotate_max)
        
        
        self.SBModelInputSize_x.SpinBox.valueChanged.connect(self.size)
        self.SBModelInputSize_y.SpinBox.valueChanged.connect(self.size)
        
        self.CBFliplr.clicked.connect(self.flip)
        self.CBFliptb.clicked.connect(self.flip)
        
        self.SBNoise.SpinBox.valueChanged.connect(self.noise)
        self.SBPieceWise.SpinBox.valueChanged.connect(self.piecewise)
        self.SBDropout.SpinBox.valueChanged.connect(self.dropout)
        
        self.SBPercentage.SpinBox.valueChanged.connect(self.affinepercentage)
        self.SBScaleMin.SpinBox.valueChanged.connect(self.scale)
        self.SBScaleMax.SpinBox.valueChanged.connect(self.scale)
        self.SBTranslateMin.SpinBox.valueChanged.connect(self.translate)
        self.SBTranslateMax.SpinBox.valueChanged.connect(self.translate)
        self.SBShearMin.SpinBox.valueChanged.connect(self.shear)
        self.SBShearMax.SpinBox.valueChanged.connect(self.shear)
        self.SBRotMin.SpinBox.valueChanged.connect(self.rotate)
        self.SBRotMax.SpinBox.valueChanged.connect(self.rotate)
        
        
        self.augmentations.setChecked(self.parent.dl.augmentation.enableAll)
        self.augmentations.clicked.connect(self.enableAll)
        
        self.Affine.setChecked(self.parent.dl.augmentation.enableAffine)
        self.Affine.clicked.connect(self.enableAffine)
        
    def enableAll(self):
        self.parent.dl.augmentation.enableAll = self.augmentations.isChecked()
    
    def size(self):
        self.parent.dl.augmentation.outputwidth = self.SBModelInputSize_x.SpinBox.value()
        self.parent.dl.augmentation.outputheight = self.SBModelInputSize_x.SpinBox.value()
        
    def flip(self):
        self.parent.dl.augmentation.flip_horz = self.CBFliplr.isChecked()
        self.parent.dl.augmentation.flip_vert = self.CBFliptb.isChecked()
        
    def noise(self):
        self.parent.dl.augmentation.prob_gaussian = self.SBNoise.SpinBox.value() /100

    def piecewise(self):
        self.parent.dl.augmentation.prob_piecewise = self.SBPieceWise.SpinBox.value() /100
        
    def dropout(self):
        self.parent.dl.augmentation.prob_dropout = self.SBDropout.SpinBox.value() /100
        
    def enableAffine(self):
        self.parent.dl.augmentation.enableAffine = self.Affine.isChecked()
        
    def affinepercentage(self):
        self.parent.dl.augmentation.prob_affine = self.SBPercentage.SpinBox.value() /100
        
    def scale(self):
        self.parent.dl.augmentation.scale_min = self.SBScaleMin.SpinBox.value()
        self.parent.dl.augmentation.scale_max = self.SBScaleMax.SpinBox.value()

    def translate(self):
        self.parent.dl.augmentation.translate_min = self.SBTranslateMin.SpinBox.value()
        self.parent.dl.augmentation.translate_max = self.SBTranslateMax.SpinBox.value()
        
    def shear(self):
        self.parent.dl.augmentation.shear_min = self.SBShearMin.SpinBox.value()
        self.parent.dl.augmentation.shear_max = self.SBShearMax.SpinBox.value()
    
    def rotate(self):
        self.parent.dl.augmentation.rotate_min = self.SBRotMin.SpinBox.value()
        self.parent.dl.augmentation.rotate_max = self.SBRotMax.SpinBox.value()