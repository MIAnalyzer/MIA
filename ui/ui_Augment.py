# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.ui_utils import LabelledSpinBox
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


        self.CBEnableAll = QCheckBox("Enable Augmentation",self.centralWidget)
        self.CBEnableAll.setChecked(True)
        self.CBEnableAll.setToolTip('Check enable/disable all augmentaions')
        self.vlayout.addWidget(self.CBEnableAll)


        self.augmentations = self.AugmentationGroup()
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
        layout.addWidget(self.Affine)

        groupBox.setLayout(layout)

        return groupBox


    def FlipGroup(self):
        groupBox = QGroupBox("Flip")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.CBFliplr = QCheckBox("Flip horizontal",self.centralWidget)
        self.CBFliplr.setChecked(True)
        self.CBFliplr.setToolTip('Check to flip images horizontally')
        self.CBFliptb = QCheckBox("Flip vertical",self.centralWidget)
        self.CBFliptb.setChecked(True)
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
        self.CBNoise = LabelledSpinBox(' Noise', self.centralWidget)
        self.CBNoise.SpinBox.setRange(0,100)
        self.CBNoise.setToolTip('Percentage to apply gaussian noise to the input image')

        self.CBPieceWise = LabelledSpinBox(' Piecewise affine', self.centralWidget)
        self.CBPieceWise.SpinBox.setRange(0,100)
        self.CBPieceWise.setToolTip('Percentage to apply piecewise affine transformation')

        self.CBDropout = LabelledSpinBox(' dropout', self.centralWidget)
        self.CBDropout.SpinBox.setRange(0,100)
        self.CBDropout.setToolTip('Percentage to apply dropout')

        layout.addWidget(self.CBNoise)
        layout.addWidget(self.CBPieceWise)
        layout.addWidget(self.CBDropout)

        groupBox.setLayout(layout)
        return groupBox

    def AffineGroup(self):

        groupBox = QGroupBox("Affine")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(1)
        layout.setContentsMargins(1, 1, 1, 1)
        hlayout = QHBoxLayout(self.centralWidget)
        self.CBEnableAffine = QCheckBox("Enable Affine ",self.centralWidget)
        self.CBEnableAffine.setChecked(True)
        self.CBEnableAffine.setToolTip('Check enable/disable affine transformatio')
        self.SBPercentage = LabelledSpinBox(' Percentage', self.centralWidget)
        self.SBPercentage.setToolTip('Percentage to apply affine transformation')
        self.SBPercentage.SpinBox.setRange(0,100)
        hlayout.addWidget(self.CBEnableAffine)
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
        self.SBScaleMin = LabelledSpinBox(' Scale Min', self.centralWidget)
        self.SBScaleMin.setToolTip('Set scale maximum percentage')
        self.SBScaleMin.SpinBox.setRange(0,1024)

        self.SBScaleMax = LabelledSpinBox(' Scale Max', self.centralWidget)
        self.SBScaleMax.setToolTip('Set scale minimum percentage')
        self.SBScaleMax.SpinBox.setRange(0,1024)

        layout.addWidget(self.SBScaleMin)
        layout.addWidget(self.SBScaleMax)


        groupBox.setLayout(layout)
        return groupBox

    def TranslateGroup(self):
        groupBox = QGroupBox("Translate")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBTranslateMin = LabelledSpinBox(' Translate Min', self.centralWidget)
        self.SBTranslateMin.setToolTip('Set translation minimun percent')
        self.SBTranslateMin.SpinBox.setRange(0,1024)

        self.SBTranslateMax = LabelledSpinBox(' Translate Max', self.centralWidget)
        self.SBTranslateMax.setToolTip('Set translation minimun percent')
        self.SBTranslateMax.SpinBox.setRange(0,1024)

        layout.addWidget(self.SBTranslateMin)
        layout.addWidget(self.SBTranslateMax)

        groupBox.setLayout(layout)
        return groupBox

    def ShearGroup(self):
        groupBox = QGroupBox("Shear")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBShearMin = LabelledSpinBox(' Shear Min', self.centralWidget)
        self.SBShearMin.setToolTip('Set shear minimum')
        self.SBShearMin.SpinBox.setRange(0,1024)

        self.SBShearMax = LabelledSpinBox(' Shear Max', self.centralWidget)
        self.SBShearMax.setToolTip('Set shear maximum')
        self.SBShearMax.SpinBox.setRange(0,1024)

        layout.addWidget(self.SBShearMin)
        layout.addWidget(self.SBShearMax)

        groupBox.setLayout(layout)
        return groupBox

    def RotateGroup(self):
        groupBox = QGroupBox("Rotate")
        layout = QVBoxLayout(self.centralWidget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.SBRotMin = LabelledSpinBox(' Rotation Min', self.centralWidget)
        self.SBRotMin.setToolTip('Set rotation range minimum')
        self.SBRotMin.SpinBox.setRange(-45,0)

        self.SBRotMax = LabelledSpinBox(' Rotation Max', self.centralWidget)
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
        