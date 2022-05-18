# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:28:44 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from ui.ui_utils import ClassList, ClassWidget, DCDButton, LabelledSpinBox
from ui.style import styleForm, getHighlightColor, getBackgroundColor, getBrightColor

class MainWindow(object):
    def setupUi(self, Form):
        screen_resolution = QDesktopWidget().screenGeometry()


        height = int(0.85*screen_resolution.height())
        width = int(height * 4/3)

        Form.setWindowTitle('Microscopic Image Analyzer') 
        Form.setWindowIcon(QIcon('icons/logo.png'))
        Form.resize(width, height)
        Form.setStyleSheet("background: " + getBackgroundColor(asString = True))
        styleForm(Form)


        self.centralWidget = QWidget(Form)

        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        
        ###### menu
        self.menubar = QMenuBar(self.centralWidget)
        
        self.FileMenu = self.menubar.addMenu("File")
        self.ASetTrainingFolder = QAction("Training Folder", self.centralWidget)
        self.FileMenu.addAction(self.ASetTrainingFolder)
        

        self.ASetTestFolder = QAction("Test Folder", self.centralWidget)
        self.FileMenu.addAction(self.ASetTestFolder)
        
        self.ASaveSettings = QAction("Save Settings", self.centralWidget)
        self.FileMenu.addAction(self.ASaveSettings)
        self.ALoadSettings = QAction("Load Settings", self.centralWidget)
        self.FileMenu.addAction(self.ALoadSettings)
        
        self.ASaveDeepLearning = QAction("Save DL Object", self.centralWidget)
        self.FileMenu.addAction(self.ASaveDeepLearning)
        
        
        
        self.FileMenu.addSeparator()
        self.AExit = QAction("Quit", self)
        self.FileMenu.addAction(self.AExit)
        self.EditMenu = self.menubar.addMenu("Edit")
        self.ASettings = QAction("Settings")
        self.EditMenu.addAction(self.ASettings)
        self.AboutMenu = self.menubar.addMenu("About")
        self.AOpenManual = QAction("Manual", self.centralWidget)
        self.AOpenGithub = QAction("Source Code", self.centralWidget)
        self.AOpenLog = QAction("Show Log", self.centralWidget)
        self.AboutMenu.addAction(self.AOpenManual)
        self.AboutMenu.addAction(self.AOpenGithub)
        self.AboutMenu.addAction(self.AOpenLog)
        
        
        self.setMenuBar(self.menubar)
        self.hlayout = QHBoxLayout()
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
        self.CBLearningMode.addItem("Object Detection")
        self.CBLearningMode.addItem("Segmentation")
        self.CBLearningMode.setToolTip('Select analysis mode')        
        self.CBLearningMode.setObjectName("nn_Learning_Mode")
        self.CBLearningMode.setEditable(False)
        self.CBLearningMode.setFocus(False)
        self.vlayout.addWidget(self.CBLearningMode)
        
        bwidth = 28
        bheight = 28
        
        self.TrainImagelayout = QHBoxLayout()
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
        
        self.TestImagelayout = QHBoxLayout()
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

        self.TVFiles = QTreeView(self.centralWidget)
        self.TVFiles.setIndentation(8)
        self.TVFiles.setVisible(False)
        self.vlayout.addWidget(self.TVFiles)
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
        self.vlayout.addWidget(line)
        
        
        # settings for segmentation
        self.SegmentationSettingslayout = QHBoxLayout()
        self.SegmentationSettingslayout.setContentsMargins(0, 0, 0, 0)
        layout0 = QVBoxLayout()
        self.Bautoseg = DCDButton(self.centralWidget)
        self.Bautoseg.setToolTip('press to perform auto segmentation')
        self.Bautoseg.setText('Auto Seg')
        self.Bautoseg.setIcon(QIcon('icons/magicwand.png'))
        
        self.Bassist = DCDButton(self.centralWidget)
        self.Bassist.setText('DEXTR')
        self.Bassist.setObjectName('assist')
        self.Bassist.setToolTip('Select assisted extreme point segmentation. Mark extreme points (top, left, right, bottom) of target object.')
        self.Bassist.setIcon(QIcon('icons/dextr.png'))
        layout0.addWidget(self.Bautoseg)
        layout0.addWidget(self.Bassist)

        layout = QVBoxLayout()
        self.CBSmartMode = QCheckBox("Smart Mode",self.centralWidget)
        self.CBSmartMode.setToolTip("Check to activate smart segmentation mode")
        self.CBSmartMode.setObjectName('SmartMode')
        self.CBInner = QCheckBox("Inner Contours",self.centralWidget)
        self.CBInner.setToolTip("Check to allow contours to have holes")
        self.CBInner.setObjectName('InnerContours')
        self.CBInner.setChecked(True)
        layout.addWidget(self.CBSmartMode)
        layout.addWidget(self.CBInner)

        self.SegmentationSettingslayout.addLayout(layout)
        self.SegmentationSettingslayout.addLayout(layout0)
        self.SegmentationSettings = QFrame()
        self.SegmentationSettings.setLayout(self.SegmentationSettingslayout)
        self.SegmentationSettings.hide()
        self.vlayout.addWidget(self.SegmentationSettings)
        
        
        # settings for object detection
        self.DetectionSettingslayout = QHBoxLayout()
        self.DetectionSettingslayout.setContentsMargins(0, 0, 0, 0)
        self.Bautodect = DCDButton(self.centralWidget)
        self.Bautodect.setToolTip('press to perform auto detection')
        self.Bautodect.setText('Auto Detection')
        self.Bautodect.setIcon(QIcon('icons/magicwand.png'))
        
        self.DetectionSettingslayout.addWidget(self.Bautodect)
        self.DetectionSettings = QFrame()
        self.DetectionSettings.setLayout(self.DetectionSettingslayout)
        self.DetectionSettings.hide()
        self.vlayout.addWidget(self.DetectionSettings)
        

        
        # tools:
        # drag tool
        self.Bdrag = DCDButton(self.centralWidget)
        self.Bdrag.setToolTip('Select drag tool')
        self.Bdrag.setVisible(False)
        self.Bdrag.setObjectName('drag')
        self.Bdrag.setMaximumSize(bwidth, bheight)
        self.Bdrag.setIcon(QIcon('icons/drag.png'))
        #draw tool
        self.Bdraw = DCDButton(self.centralWidget)
        self.Bdraw.setMaximumSize(bwidth, bheight)
        self.Bdraw.setVisible(False)
        self.Bdraw.setObjectName('draw')
        self.Bdraw.setToolTip('Select drawing tool')
        self.Bdraw.setIcon(QIcon('icons/draw.png'))
        # settings for drawing tool
        self.DrawSettingslayout = QHBoxLayout()
        self.DrawSettingslayout.setContentsMargins(0, 0, 0, 0)
        self.CBAddShape = QRadioButton(self.centralWidget)
        self.CBAddShape.setText("Add ")
        self.CBAddShape.setChecked(True)
        self.CBAddShape.setToolTip('Toggle to add shape')
        self.CBDelShape = QRadioButton(self.centralWidget)
        self.CBDelShape.setText("Delete  ")
        self.CBDelShape.setToolTip('Toggle to delete shape')
        self.CBSliceShape = QRadioButton(self.centralWidget)
        self.CBSliceShape.setText("Slice  ")
        self.CBSliceShape.setToolTip('Toggle to slice shapes')
        self.DrawSettingslayout.addWidget(self.CBAddShape)
        self.DrawSettingslayout.addWidget(self.CBDelShape)
        self.DrawSettingslayout.addWidget(self.CBSliceShape)
        self.DrawSettings = QFrame()
        self.DrawSettings.setLayout(self.DrawSettingslayout)
        self.DrawSettings.hide()
        # polyline tool
        self.Bpoly = DCDButton(self.centralWidget)
        self.Bpoly.setMaximumSize(bwidth, bheight)
        self.Bpoly.setVisible(False)
        self.Bpoly.setObjectName('poly')
        self.Bpoly.setToolTip('Select polygon tool')
        self.Bpoly.setIcon(QIcon('icons/poly.png'))
        # assign tool
        self.Bassign = DCDButton(self.centralWidget)
        self.Bassign.setMaximumSize(bwidth, bheight)
        self.Bassign.setVisible(False)
        self.Bassign.setObjectName('assign')
        self.Bassign.setToolTip('Select assign class tool')
        self.Bassign.setIcon(QIcon('icons/assign.png'))
        # extend tool
        self.Bextend = DCDButton(self.centralWidget)
        self.Bextend.setMaximumSize(bwidth, bheight)
        self.Bextend.setVisible(False)
        self.Bextend.setObjectName('extend')
        self.Bextend.setToolTip('Select extend/erase tool')
        self.Bextend.setIcon(QIcon('icons/expand.png'))
        # settings for extend tool
        self.ExtendSettingslayout = QHBoxLayout()
        self.ExtendSettingslayout.setContentsMargins(0, 0, 0, 0)
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
        self.SSize.setMaximumWidth((self.CBExtend.width()+self.CBErase.width())/4)
        self.ExtendSettingslayout.addWidget(self.CBExtend)
        self.ExtendSettingslayout.addWidget(self.CBErase)
        self.ExtendSettingslayout.addWidget(self.SSize)
        self.ExtendSettings = QFrame()
        self.ExtendSettings.setLayout(self.ExtendSettingslayout)
        self.ExtendSettings.hide()
        # delete tool
        self.Bdelete = DCDButton(self.centralWidget)
        self.Bdelete.setMaximumSize(bwidth, bheight)
        self.Bdelete.setVisible(False)
        self.Bdelete.setObjectName('delete')
        self.Bdelete.setToolTip('Select delete tool')
        self.Bdelete.setIcon(QIcon('icons/delete.png'))
        # set object tool
        self.BsetObject = DCDButton(self.centralWidget)
        self.BsetObject.setObjectName('point')
        self.BsetObject.setVisible(False)
        self.BsetObject.setMaximumSize(bwidth, bheight)
        self.BsetObject.setToolTip('Select set object position tool')
        self.BsetObject.setIcon(QIcon('icons/addobject.png'))       
        # shift object tool
        self.BshiftObject = DCDButton(self.centralWidget)
        self.BshiftObject.setMaximumSize(bwidth, bheight)
        self.BshiftObject.setVisible(False)
        self.BshiftObject.setObjectName('shift')
        self.BshiftObject.setToolTip('Select shift object tool')
        self.BshiftObject.setIcon(QIcon('icons/shift.png'))
        # assign class tool
        self.BassignClass = DCDButton(self.centralWidget)
        self.BassignClass.setMaximumSize(bwidth, bheight)
        self.BassignClass.setObjectName('assignimageclass')
        self.BassignClass.setToolTip('Assign current class')
        self.BassignClass.setIcon(QIcon('icons/assignclass.png'))
        # set class tool
        self.BsetClass = DCDButton(self.centralWidget)
        self.BsetClass.setMaximumSize(bwidth, bheight)
        self.BsetClass.setObjectName('setimageclass')
        self.BsetClass.setToolTip('Assign current class and continue to next image')
        self.BsetClass.setIcon(QIcon('icons/setclass.png'))
        # set all to class tool
        self.BsetallClass = DCDButton(self.centralWidget)
        self.BsetallClass.setMaximumSize(bwidth, bheight)
        self.BsetallClass.setObjectName('setallimagesclass')
        self.BsetallClass.setToolTip('Assign current class to all images in the current folder')
        self.BsetallClass.setIcon(QIcon('icons/setallclass.png'))

        
        # tool buttons
        self.ToolButtonslayout = QHBoxLayout()
        self.ToolButtonslayout.setContentsMargins(0, 0, 0, 0)
        self.ToolButtons = QFrame()
        self.ToolButtons.setMaximumWidth(self.CBExtend.width()*3)
        self.ToolButtons.setLayout(self.ToolButtonslayout)
        self.ToolButtonslayout.addWidget(self.Bdrag)
        self.ToolButtonslayout.addWidget(self.Bdraw)
        self.ToolButtonslayout.addWidget(self.BsetObject)
        self.ToolButtonslayout.addWidget(self.Bpoly)
        self.ToolButtonslayout.addWidget(self.BshiftObject)
        self.ToolButtonslayout.addWidget(self.Bassign)
        self.ToolButtonslayout.addWidget(self.Bextend)
        self.ToolButtonslayout.addWidget(self.Bdelete)

        self.ClassificationToolButtonslayout = QHBoxLayout()
        self.ClassificationToolButtonslayout.setContentsMargins(0, 0, 0, 0)
        self.ClassificationToolButtons = QFrame()
        self.ClassificationToolButtons.setLayout(self.ClassificationToolButtonslayout)
        self.ClassificationToolButtons.hide()
        self.ClassificationToolButtonslayout.addWidget(self.BassignClass)
        self.ClassificationToolButtonslayout.addWidget(self.BsetClass)
        self.ClassificationToolButtonslayout.addWidget(self.BsetallClass)
        
        self.vlayout.addWidget(self.ClassificationToolButtons)
        self.vlayout.addWidget(self.ToolButtons)

        
        # add settings frames
        self.vlayout.addWidget(self.DrawSettings)
        self.vlayout.addWidget(self.ExtendSettings)

        
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
        self.vlayout.addWidget(line)

        # tracking mode
        self.Trackinglayout = QVBoxLayout()
        self.Trackinglayout.setContentsMargins(0, 0, 0, 0)
        self.Tracking = QFrame()
        self.Tracking.setLayout(self.Trackinglayout)

        self.BTrack = DCDButton(self.centralWidget, 'Calculate Tracking')
        self.BTrack.setToolTip('Press to calculate tracking')
        self.BTrack.setIcon(QIcon('icons/tracking.png'))

        self.trcolorlayout = QHBoxLayout()
        self.CBtrackcolor = QCheckBox("Color Coding",self.centralWidget)
        self.CBtrackcolor.setToolTip("Check to change coloring by objects, uncheck to color by classes")
        self.CBtrackcolor.setObjectName('TrackColor')
        self.CBtrackcolor.setChecked(True)

        self.CBshowtrack = QCheckBox("Show Track",self.centralWidget)
        self.CBshowtrack.setToolTip("Check to show track")
        self.CBshowtrack.setObjectName('ShowTrack')
        self.CBshowtrack.setChecked(True)
        self.trcolorlayout.addWidget(self.CBtrackcolor)
        self.trcolorlayout.addWidget(self.CBshowtrack)

        self.objNumLayout = QHBoxLayout()
        self.BSetObjectNumber = DCDButton(self.centralWidget)
        self.BSetObjectNumber.setMaximumSize(bwidth, bheight)
        self.BSetObjectNumber.setObjectName('objectnumber')
        self.BSetObjectNumber.setIcon(QIcon('icons/objectnumber.png'))
        self.BSetObjectNumber.setToolTip('Select set object number tool')

        self.BDeleteObject = DCDButton(self.centralWidget)
        self.BDeleteObject.setMaximumSize(bwidth, bheight)
        self.BDeleteObject.setObjectName('deleteobject')
        self.BDeleteObject.setIcon(QIcon('icons/delete.png'))
        self.BDeleteObject.setToolTip('Select delete object tool')

        self.BChangeObjectColor = DCDButton(self.centralWidget)
        self.BChangeObjectColor.setMaximumSize(bwidth, bheight)
        self.BChangeObjectColor.setObjectName('objectcolor')
        self.BChangeObjectColor.setIcon(QIcon('icons/objectcolor.png'))
        self.BChangeObjectColor.setToolTip('Select change color tool')

        self.objNumLayout.addWidget(self.BSetObjectNumber)
        self.objNumLayout.addWidget(self.BChangeObjectColor)
        self.objNumLayout.addWidget(self.BDeleteObject)

        self.Trackinglayout.addWidget(self.BTrack)
        self.Trackinglayout.addLayout(self.trcolorlayout)
        self.Trackinglayout.addLayout(self.objNumLayout)

        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
        self.Trackinglayout.addWidget(line)

        self.Tracking.setVisible(False)
        self.vlayout.addWidget(self.Tracking)


        
        self.ralayout = QHBoxLayout()
        self.Bundo = DCDButton(self.centralWidget, "Undo")
        self.Bundo.setIcon(QIcon('icons/reset.png'))
        self.Bundo.setToolTip('Undo drawing') 
        self.Bclear = DCDButton(self.centralWidget, "Clear Image")
        self.Bclear.setIcon(QIcon('icons/clear.png'))
        self.Bclear.setToolTip('Reset image')  
        self.ralayout.addWidget(self.Bundo) 
        self.ralayout.addWidget(self.Bclear)    
        self.vlayout.addItem(self.ralayout) 
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
        self.vlayout.addWidget(line)

        self.trlayout = QHBoxLayout()
        self.Btrain = DCDButton(self.centralWidget, "Train Model")
        self.Btrain.setToolTip('Open training window')
        self.Btrain.setIcon(QIcon('icons/train.png'))

                
        self.Bresetmodel = DCDButton(self.centralWidget, "Reset Model")
        self.Bresetmodel.setIcon(QIcon('icons/clear.png'))
        self.Bresetmodel.setToolTip('reset trained model')

        self.trlayout.addWidget(self.Btrain)    
        self.trlayout.addWidget(self.Bresetmodel)    
        self.vlayout.addItem(self.trlayout) 
            

        
        self.predictlayout = QHBoxLayout()
        self.Bpredictall = DCDButton(self.centralWidget, "Predict All")
        self.Bpredictall.setIcon(QIcon('icons/predictall.png'))
        self.Bpredictall.setToolTip('Predict all images in prediction folder')
        self.Bpredict = DCDButton(self.centralWidget, "Predict Current")
        self.Bpredict.setIcon(QIcon('icons/predict.png'))
        self.Bpredict.setToolTip('Predict current field of view')
        self.predictlayout.addWidget(self.Bpredictall)    
        self.predictlayout.addWidget(self.Bpredict)
        
        self.vlayout.addItem(self.predictlayout) 
        
        self.lslayout = QHBoxLayout()
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
        
        
        
        

        
        self.add_del_layout = QHBoxLayout()
        self.Baddclass = DCDButton(self.centralWidget)
        self.Baddclass.setIcon(QIcon('icons/addobject.png'))
        self.Baddclass.setToolTip('Add a new class')
        self.Baddclass.setMaximumHeight(bheight)
        self.Baddclass.setStyleSheet('text-align:center')
        self.Baddclass.resetStyle()      
        self.add_del_layout.addWidget(self.Baddclass)
        self.Bdelclass = DCDButton(self.centralWidget)
        self.Bdelclass.setIcon(QIcon('icons/delete.png'))
        self.Bdelclass.setToolTip('Delete class')
        self.Bdelclass.setMaximumHeight(bheight)
        self.Bdelclass.setStyleSheet('text-align:center')
        self.Bdelclass.resetStyle()  
        self.add_del_layout.addWidget(self.Bdelclass)
        self.vlayout.addItem(self.add_del_layout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
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
        
        self.CBKeep = QCheckBox("Keep Settings",self.centralWidget)
        self.CBKeep.setToolTip("Check to keep brightness/contrast and fov settings (recommended for time series)")
        self.CBKeep.setObjectName('KeepImageSettings')
        self.CBKeep.setChecked(False)
        self.vlayout.addWidget(self.CBKeep)

        brightnesslayout = QHBoxLayout()
        self.SBrightness = QSlider(Qt.Horizontal, self.centralWidget)
        self.SBrightness.setMinimum(-200)
        self.SBrightness.setMaximum(200)
        self.SBrightness.setValue(0)
        self.SBrightness.setToolTip('Change Image Brightness')
        self.LBrightness = QLabel(self.centralWidget)
        self.LBrightness.setText(' Brightness')
        brightnesslayout.addWidget(self.SBrightness, stretch = 2)
        brightnesslayout.addWidget(self.LBrightness, stretch = 1)
        
        contrastlayout = QHBoxLayout()
        self.SContrast = QSlider(Qt.Horizontal, self.centralWidget)
        self.SContrast.setMinimum(-50)
        self.SContrast.setMaximum(250)
        self.SContrast.setValue(0)
        self.SContrast.setToolTip('Change Image Contrast')
        self.LContrast = QLabel(self.centralWidget)
        self.LContrast.setText(' Contrast')
        contrastlayout.addWidget(self.SContrast, stretch = 2)
        contrastlayout.addWidget(self.LContrast, stretch = 1)
        
        self.vlayout.addItem(brightnesslayout)
        self.vlayout.addItem(contrastlayout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background: " + getBrightColor(asString = True))
        self.vlayout.addWidget(line)
        
        self.nextprev_layout = QHBoxLayout()
        self.Bprev = DCDButton(self.centralWidget)
        self.Bprev.setIcon(QIcon('icons/previous.png'))
        self.Bprev.setStyleSheet('text-align:center')
        self.Bprev.setToolTip('Previous image')
        self.Bprev.resetStyle()   
        self.nextprev_layout.addWidget(self.Bprev)
        self.Bnext = DCDButton(self.centralWidget)
        self.Bnext.setIcon(QIcon('icons/next.png'))
        self.Bnext.setStyleSheet('text-align:center')
        self.Bnext.setToolTip('Next image')
        self.Bnext.resetStyle()  
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
        self.stacklayout = QVBoxLayout()
        self.SFrame = QSlider(Qt.Vertical, self.centralWidget)
        self.SFrame.setMinimum(1)
        self.SFrame.setMaximum(10)
        self.SFrame.setValue(1)
        self.SFrame.setToolTip('Change Frame of Stack')
        self.BCopyStackLabel = DCDButton(self.centralWidget)
        self.BCopyStackLabel.setMaximumSize(bwidth//1.5, bheight//1.5)       
        self.BCopyStackLabel.setToolTip('Press to reuse current label for all frames in the current stack')
        self.BCopyStackLabel.setIcon(QIcon('icons/predictall.png'))  
        self.stacklayout.addWidget(self.SFrame)
        self.stacklayout.addWidget(self.BCopyStackLabel)
        self.StackSettings = QFrame()
        self.StackSettings.setLayout(self.stacklayout)
        self.StackSettings.hide()
        
        
        
        self.hlayout.addWidget(self.StackSettings)
        
        
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
        

 

        
        
        