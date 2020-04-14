# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:28:44 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ui.ui_utils import ClassList, ClassWidget, DCDButton


class MainWindow(object):
    def setupUi(self, Form):
       
        screen_resolution = QDesktopWidget().screenGeometry()
        
        height = screen_resolution.height() - 150
        width = height * 4/3
        
        
        Form.setWindowTitle('DeepCellDetector') 
        Form.setWindowIcon(QIcon('icons/logo.png'))
        Form.resize(width, height)
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")
        
        self.centralWidget = QWidget(Form)
        
        
        
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        
        ###### menu
        self.menubar = QMenuBar(self.centralWidget)
        
        self.FileMenu = self.menubar.addMenu("File")
        self.FileMenu.setStyleSheet("QMenu:item:selected { background:#3399ff; color:white;}")
        self.ASetTrainingFolder = QAction("Training Folder", self.centralWidget)
        self.FileMenu.addAction(self.ASetTrainingFolder)
        
        self.ASetTestFolder = QAction("Test Folder", self.centralWidget)

        self.FileMenu.addAction(self.ASetTestFolder)
        
        self.FileMenu.addSeparator()
        self.AExit = QAction("Quit", self)
        self.FileMenu.addAction(self.AExit)
        
        self.EditMenu = self.menubar.addMenu("Edit")
        self.ASettings = QAction("Settings")
        self.EditMenu.setStyleSheet("QMenu:item:selected { background:#3399ff; color:white;}")
        self.EditMenu.addAction(self.ASettings)
        
        self.AboutMenu = self.menubar.addMenu("About")
        self.AboutMenu.setStyleSheet("QMenu:item:selected { background:#3399ff; color:white;}")
        
        self.setMenuBar(self.menubar)

        self.hlayout = QHBoxLayout(self.centralWidget)
        self.hlayout.setContentsMargins(3, 3, 3, 3)
        self.hlayout.setSpacing(6)
        self.centralWidget.setLayout(self.hlayout)
        
        
        self.vlayout = QVBoxLayout()
        verticalSpacer = QSpacerItem(20, 40)
        self.vlayout.addItem(verticalSpacer)
        
        
        ## vertical layout - buttons
        self.BSettings = DCDButton(self.centralWidget, 'Settings')
        self.BSettings.setToolTip('Open settings sindow')
        self.BSettings.setIcon(QIcon('icons/settings.png'))
        self.BSettings.setFlat(True)

        self.vlayout.addWidget(self.BSettings)
        
        
        self.CBLearningMode = QComboBox(self.centralWidget)
        self.CBLearningMode.addItem("Classification")
        self.CBLearningMode.addItem("Segmentation")
        self.CBLearningMode.addItem("Time Series")
        self.CBLearningMode.setToolTip('Select analysis mode')
        self.CBLearningMode.setObjectName("CBmode")
        self.CBLearningMode.setEditable(False)
        self.CBLearningMode.setFocus(False)
        self.vlayout.addWidget(self.CBLearningMode)
        
        
        bwidth = 28
        bheight = 28
        
        self.TrainImagelayout = QHBoxLayout(self.centralWidget)
        self.BLoadTrainImages = DCDButton(self.centralWidget)
        self.BLoadTrainImages.setToolTip('Set folder with training images')
        self.BLoadTrainImages.setMaximumSize(bwidth, bheight)
        self.BLoadTrainImages.setIcon(QIcon('icons/load.png'))

        self.BTrainImageFolder = DCDButton(self.centralWidget, "Set Training Image Folder")
        self.BTrainImageFolder.setToolTip('Training image folder')
        self.BTrainImageFolder.setEnabled(False)
        self.TrainImagelayout.addWidget(self.BLoadTrainImages)
        self.TrainImagelayout.addWidget(self.BTrainImageFolder)
        self.vlayout.addItem(self.TrainImagelayout)
        
        self.TestImagelayout = QHBoxLayout(self.centralWidget)
        self.BLoadTestImages = DCDButton(self.centralWidget)
        self.BLoadTestImages.setMaximumSize(bwidth, bheight)
        self.BLoadTestImages.setToolTip('Set folder with prediction images')
        self.BLoadTestImages.setIcon(QIcon('icons/load.png'))

        self.BTestImageFolder = DCDButton(self.centralWidget,"Set Test Image Folder")
        self.BTestImageFolder.setToolTip('Prediction image folder')
        self.BTestImageFolder.setEnabled(False)
        self.TestImagelayout.addWidget(self.BLoadTestImages)
        self.TestImagelayout.addWidget(self.BTestImageFolder)
        self.vlayout.addItem(self.TestImagelayout)
        

        ### segmentation
        self.SButtonslayout = QHBoxLayout(self.centralWidget)
        self.SButtonslayout.setContentsMargins(0, 0, 0, 0)
        self.Bdrag = DCDButton(self.centralWidget)
        self.Bdrag.setToolTip('Select drag tool')
        self.Bdrag.setObjectName('drag')
        self.Bdrag.setMaximumSize(bwidth, bheight)
        self.Bdrag.setIcon(QIcon('icons/drag.png'))
        self.SButtonslayout.addWidget(self.Bdrag)

        self.Bdraw = DCDButton(self.centralWidget)
        self.Bdraw.setMaximumSize(bwidth, bheight)
        self.Bdraw.setObjectName('draw')
        self.Bdraw.setToolTip('Select drawing tool')
        self.Bdraw.setIcon(QIcon('icons/draw.png'))
        self.SButtonslayout.addWidget(self.Bdraw)
        
        self.Bpoly = DCDButton(self.centralWidget)
        self.Bpoly.setMaximumSize(bwidth, bheight)
        self.Bpoly.setObjectName('poly')
        self.Bpoly.setToolTip('Select polygon tool')
        self.Bpoly.setIcon(QIcon('icons/poly.png'))
        self.SButtonslayout.addWidget(self.Bpoly)

        self.Bassign = DCDButton(self.centralWidget)
        self.Bassign.setMaximumSize(bwidth, bheight)
        self.Bassign.setObjectName('assign')
        self.Bassign.setToolTip('Select assign class tool')
        self.Bassign.setIcon(QIcon('icons/assign.png'))
        self.SButtonslayout.addWidget(self.Bassign)
        
        self.Bextend = DCDButton(self.centralWidget)
        self.Bextend.setMaximumSize(bwidth, bheight)
        self.Bextend.setObjectName('extend')
        self.Bextend.setToolTip('Select extend/erase tool')
        self.Bextend.setIcon(QIcon('icons/expand.png'))
        self.SButtonslayout.addWidget(self.Bextend)
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.Bdelete = DCDButton(self.centralWidget)
        self.Bdelete.setMaximumSize(bwidth, bheight)
        self.Bdelete.setObjectName('delete')
        self.Bdelete.setToolTip('Select delete shape tool')
        self.Bdelete.setIcon(QIcon('icons/delete.png'))
        self.SButtonslayout.addWidget(self.Bdelete)

        self.SegmentButtons = QFrame()
        self.SegmentButtons.setLayout(self.SButtonslayout)
        
        self.vlayout.addWidget(self.SegmentButtons)
        

        self.DrawSettingslayout = QHBoxLayout(self.centralWidget)
        self.CBAddShape = QRadioButton(self.centralWidget)
        self.CBAddShape.setText("Add ")
        self.CBAddShape.setChecked(True)
        self.CBAddShape.setToolTip('Toggle to add shape')
        self.CBDelShape = QRadioButton(self.centralWidget)
        self.CBDelShape.setText("Delete  ")
        self.CBDelShape.setToolTip('Toggle to delete shape')
        self.DrawSettingslayout.addWidget(self.CBAddShape)
        self.DrawSettingslayout.addWidget(self.CBDelShape)
        self.DrawSettings = QFrame()
        self.DrawSettings.setLayout(self.DrawSettingslayout)
        self.DrawSettings.hide()
        self.vlayout.addWidget(self.DrawSettings)

        self.ExtendSettingslayout = QHBoxLayout(self.centralWidget)
        self.CBExtend = QRadioButton(self.centralWidget)
        self.CBExtend.setText("Expand ")
        self.CBExtend.setChecked(True)
        self.CBExtend.setToolTip('Toggle to extend tool')
        self.CBErase = QRadioButton(self.centralWidget)
        self.CBErase.setText("Erase  ")
        self.CBErase.setToolTip('Toggle  to erase tool')
        self.SSize = QSlider(Qt.Horizontal, self.centralWidget)
        self.SSize.setMinimum(1)
        self.SSize.setMaximum(40)
        self.SSize.setValue(10)
        self.SSize.setToolTip('Set marker size')
        self.ExtendSettingslayout.addWidget(self.CBExtend)
        self.ExtendSettingslayout.addWidget(self.CBErase)
        self.ExtendSettingslayout.addWidget(self.SSize)
        self.ExtendSettings = QFrame()
        self.ExtendSettings.setLayout(self.ExtendSettingslayout)
        self.ExtendSettings.hide()
        self.vlayout.addWidget(self.ExtendSettings)
        
        
        ### classification
        self.CButtonslayout = QHBoxLayout(self.centralWidget)
        self.CButtonslayout.setContentsMargins(0, 0, 0, 0)
        self.BsetClass = DCDButton(self.centralWidget)
        self.BsetClass.setMaximumSize(bwidth, bheight)
        self.BsetClass.setToolTip('Set active class')
        self.BsetClass.setIcon(QIcon('icons/setclass.png'))
        self.CButtonslayout.addWidget(self.BsetClass)

        self.BassignClass = DCDButton(self.centralWidget)
        self.BassignClass.setMaximumSize(bwidth, bheight)
        self.BassignClass.setToolTip('Assign current class')
        self.BassignClass.setIcon(QIcon('icons/assignclass.png'))
        self.CButtonslayout.addWidget(self.BassignClass)
        
        self.ClassificationButtons = QFrame()
        self.ClassificationButtons.setLayout(self.CButtonslayout)
        self.vlayout.addWidget(self.ClassificationButtons)
        
        
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)

        
        self.ralayout = QHBoxLayout(self.centralWidget)
        self.Bautoseg = DCDButton(self.centralWidget, "Auto Segment")
        self.Bautoseg.setObjectName("Bautoseg")
        self.Bautoseg.setIcon(QIcon('icons/autoseg.png'))
        self.Bautoseg.setToolTip('autosegment image')
   

        self.Bclear = DCDButton(self.centralWidget, "Clear Image")
        self.Bclear.setIcon(QIcon('icons/clear.png'))
        self.Bclear.setToolTip('Reset image')


        self.ralayout.addWidget(self.Bautoseg)    
        self.ralayout.addWidget(self.Bclear)    
        self.vlayout.addItem(self.ralayout) 
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)

        self.trlayout = QHBoxLayout(self.centralWidget)
        self.Btrain = DCDButton(self.centralWidget, "Train Model")
        self.Btrain.setToolTip('Open training window')
        self.Btrain.setIcon(QIcon('icons/train.png'))

                
        self.Bresetmodel = DCDButton(self.centralWidget, "Reset Model")
        self.Bresetmodel.setIcon(QIcon('icons/resetmodel.png'))
        self.Bresetmodel.setToolTip('reset trained model')

        self.trlayout.addWidget(self.Btrain)    
        self.trlayout.addWidget(self.Bresetmodel)    
        self.vlayout.addItem(self.trlayout) 
            

        
        self.predictlayout = QHBoxLayout(self.centralWidget)
        self.Bpredictall = DCDButton(self.centralWidget, "Predict All")
        self.Bpredictall.setIcon(QIcon('icons/predict.png'))
        self.Bpredictall.setToolTip('Predict all images in prediction folder')
        self.Bpredict = DCDButton(self.centralWidget, "Predict Current")
        self.Bpredict.setIcon(QIcon('icons/predict.png'))
        self.Bpredict.setToolTip('Predict current image')
        self.predictlayout.addWidget(self.Bpredictall)    
        self.predictlayout.addWidget(self.Bpredict)
        
        self.vlayout.addItem(self.predictlayout) 
        
        self.lslayout = QHBoxLayout(self.centralWidget)
        self.Bloadmodel = DCDButton(self.centralWidget, "Load Model")
        self.Bloadmodel.setIcon(QIcon('icons/loadmodel.png'))
        self.Bloadmodel.setToolTip('Load trained model')
        
        self.Bsavemodel = DCDButton(self.centralWidget, "Save Model")
        self.Bsavemodel.setIcon(QIcon('icons/savemodel.png'))
        self.Bsavemodel.setToolTip('Save current model')
        self.lslayout.addWidget(self.Bloadmodel)    
        self.lslayout.addWidget(self.Bsavemodel)
        
        self.vlayout.addItem(self.lslayout) 
        

        
        ##############
        self.classList = ClassList(self)
        self.classList.addClass('background')
        self.classList.addClass()
        self.vlayout.addWidget(self.classList)
        
        self.add_del_layout = QHBoxLayout(self.centralWidget)
        self.Baddclass = DCDButton(self.centralWidget, "+")
        self.Baddclass.setToolTip('Add a new class')
        self.Baddclass.setMaximumHeight(bheight)
        self.Baddclass.setStyleSheet('text-align:center')
        self.add_del_layout.addWidget(self.Baddclass)
        self.Bdelclass = DCDButton(self.centralWidget, "-")
        self.Bdelclass.setToolTip('Delete class')
        self.Bdelclass.setMaximumHeight(bheight)
        self.Bdelclass.setStyleSheet('text-align:center')
        self.add_del_layout.addWidget(self.Bdelclass)
        self.vlayout.addItem(self.add_del_layout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.BPostProcessing = DCDButton(self.centralWidget, "Postprocessing")
        self.BPostProcessing.setObjectName("BPostProcessing")
        self.BPostProcessing.setIcon(QIcon('icons/postprocessing.png'))
        self.BPostProcessing.setToolTip('Open postprocessing window')
        self.vlayout.addWidget(self.BPostProcessing)
        
        self.Bresults = DCDButton(self.centralWidget, "Results")
        self.Bresults.setIcon(QIcon('icons/results.png'))
        self.Bresults.setToolTip('Open results window')
        self.vlayout.addWidget(self.Bresults)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vlayout.addItem(verticalSpacer)
        
        self.nextprev_layout = QHBoxLayout(self.centralWidget)
        self.Bprev = DCDButton(self.centralWidget, "<")
        self.Bprev.setStyleSheet('text-align:center')
        self.Bprev.setToolTip('Previous window')
        self.nextprev_layout.addWidget(self.Bprev)
        self.Bnext = DCDButton(self.centralWidget, ">")
        self.Bnext.setStyleSheet('text-align:center')
        self.Bnext.setToolTip('Next window')
        self.nextprev_layout.addWidget(self.Bnext)
        self.vlayout.addItem(self.nextprev_layout)
        
       
        ## horizontal layout - buttons + canvas    
        self.hlayout.addLayout(self.vlayout)
        horizontalspacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hlayout.addItem(horizontalspacer)
        self.canvas = QLabel(self.centralWidget)
        self.canvas.setText("")
        self.canvas.setObjectName("canvas")
        self.canvas.setPixmap(QPixmap(width, height))
        self.hlayout.addWidget(self.canvas)
        horizontalspacer = QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.hlayout.addItem(horizontalspacer)
        
        
        self.StatusFile = QLabel(self.centralWidget)
        self.StatusFile.setText('No Image Displayed')
        self.StatusFile.setObjectName("StatusFile")
        self.statusBar().addPermanentWidget(self.StatusFile, stretch = 1)
        Form.setCentralWidget(self.centralWidget)
        
        self.StatusProgress = QProgressBar(self.centralWidget)
        self.StatusProgress.setMaximumWidth(350)
        self.StatusProgress.hide()
        self.statusBar().addPermanentWidget(self.StatusProgress, stretch = 1)
        
        self.Status = QLabel(self.centralWidget)
        self.Status.setText('Initialized')
        self.Status.setObjectName("Status")
        self.statusBar().addPermanentWidget(self.Status, stretch = 0)
        

 

        
        
        