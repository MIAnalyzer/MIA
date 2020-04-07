# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:28:44 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import random



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
        self.BSettings = QPushButton(self.centralWidget)
        self.BSettings.setText('Settings')
        self.BSettings.setStyleSheet('text-align:left')
        self.BSettings.setToolTip('Open settings sindow')
        self.BSettings.setObjectName("BSettings")
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
        self.BLoadTrainImages = QPushButton(self.centralWidget)
        self.BLoadTrainImages.setObjectName("BLoadTrainImages")
        self.BLoadTrainImages.setToolTip('Set folder with training images')
        self.BLoadTrainImages.setMaximumSize(bwidth, bheight)
        self.BLoadTrainImages.setIcon(QIcon('icons/load.png'))
        self.BLoadTrainImages.setFlat(True)
        self.BTrainImageFolder = QPushButton(self.centralWidget)
        self.BTrainImageFolder.setText("Set Training Image Folder")
        self.BTrainImageFolder.setObjectName("BTrainImageFolder")
        self.BTrainImageFolder.setToolTip('Training image folder')
        self.BTrainImageFolder.setFlat(True)
        self.BTrainImageFolder.setStyleSheet('text-align:left')
        self.BTrainImageFolder.setEnabled(False)
        self.TrainImagelayout.addWidget(self.BLoadTrainImages)
        self.TrainImagelayout.addWidget(self.BTrainImageFolder)
        self.vlayout.addItem(self.TrainImagelayout)
        
        self.TestImagelayout = QHBoxLayout(self.centralWidget)
        self.BLoadTestImages = QPushButton(self.centralWidget)
        self.BLoadTestImages.setObjectName("BLoadTestImages")
        self.BLoadTestImages.setMaximumSize(bwidth, bheight)
        self.BLoadTestImages.setToolTip('Set folder with prediction images')
        self.BLoadTestImages.setIcon(QIcon('icons/load.png'))
        self.BLoadTestImages.setFlat(True)
        self.BTestImageFolder = QPushButton(self.centralWidget)
        self.BTestImageFolder.setText("Set Test Image Folder")
        self.BTestImageFolder.setObjectName("BTestImageFolder")
        self.BTestImageFolder.setFlat(True)
        self.BTestImageFolder.setToolTip('Prediction image folder')
        self.BTestImageFolder.setStyleSheet('text-align:left')
        self.BTestImageFolder.setEnabled(False)
        self.TestImagelayout.addWidget(self.BLoadTestImages)
        self.TestImagelayout.addWidget(self.BTestImageFolder)
        self.vlayout.addItem(self.TestImagelayout)
        
        
        
        ### segmentation
        self.SButtonslayout = QHBoxLayout(self.centralWidget)
        self.SButtonslayout.setContentsMargins(0, 0, 0, 0)
        self.Bdrag = QPushButton(self.centralWidget)
        self.Bdrag.setToolTip('Select drag tool')
        self.Bdrag.setMaximumSize(bwidth, bheight)
        self.Bdrag.setObjectName("Bdrag")
        self.Bdrag.setIcon(QIcon('icons/drag.png'))
        self.Bdrag.setFlat(True)
        self.SButtonslayout.addWidget(self.Bdrag)

        self.Bdraw = QPushButton(self.centralWidget)
        self.Bdraw.setMaximumSize(bwidth, bheight)
        self.Bdraw.setToolTip('Select drawing tool')
        self.Bdraw.setObjectName("Bdraw")
        self.Bdraw.setIcon(QIcon('icons/draw.png'))
        self.Bdraw.setFlat(True)
        self.SButtonslayout.addWidget(self.Bdraw)
        
        self.Bpoly = QPushButton(self.centralWidget)
        self.Bpoly.setMaximumSize(bwidth, bheight)
        self.Bpoly.setObjectName("Bpoly")
        self.Bpoly.setToolTip('Select polygon tool')
        self.Bpoly.setIcon(QIcon('icons/poly.png'))
        self.Bpoly.setFlat(True)
        self.SButtonslayout.addWidget(self.Bpoly)

        self.Bassign = QPushButton(self.centralWidget)
        self.Bassign.setMaximumSize(bwidth, bheight)
        self.Bassign.setObjectName("Bassign")
        self.Bassign.setToolTip('Select assgin class tool')
        self.Bassign.setIcon(QIcon('icons/assign.png'))
        self.Bassign.setFlat(True)
        self.SButtonslayout.addWidget(self.Bassign)
        
        self.Bextend = QPushButton(self.centralWidget)
        self.Bextend.setMaximumSize(bwidth, bheight)
        self.Bextend.setObjectName("Bexpand")
        self.Bextend.setToolTip('Select extend/erase tool')
        self.Bextend.setIcon(QIcon('icons/expand.png'))
        self.Bextend.setFlat(True)
        self.SButtonslayout.addWidget(self.Bextend)
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.Bdelete = QPushButton(self.centralWidget)
        self.Bdelete.setMaximumSize(bwidth, bheight)
        self.Bdelete.setObjectName("Bdelete")
        self.Bdelete.setToolTip('Select delete shape tool')
        self.Bdelete.setIcon(QIcon('icons/delete.png'))
        self.Bdelete.setFlat(True)
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
        self.BsetClass = QPushButton(self.centralWidget)
        self.BsetClass.setMaximumSize(bwidth, bheight)
        self.BsetClass.setObjectName("BsetClass")
        self.BsetClass.setToolTip('Set active class')
        self.BsetClass.setIcon(QIcon('icons/setclass.png'))
        self.BsetClass.setFlat(True)
        self.CButtonslayout.addWidget(self.BsetClass)

        self.BassignClass = QPushButton(self.centralWidget)
        self.BassignClass.setMaximumSize(bwidth, bheight)
        self.BassignClass.setObjectName("BassignClass")
        self.BassignClass.setToolTip('Assign current class')
        self.BassignClass.setIcon(QIcon('icons/assignclass.png'))
        self.BassignClass.setFlat(True)
        self.CButtonslayout.addWidget(self.BassignClass)
        
        self.ClassificationButtons = QFrame()
        self.ClassificationButtons.setLayout(self.CButtonslayout)
        self.vlayout.addWidget(self.ClassificationButtons)
        
        
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)

            
        self.Bclear = QPushButton(self.centralWidget)
        self.Bclear.setObjectName("Bclear")
        self.Bclear.setIcon(QIcon('icons/clear.png'))
        self.Bclear.setText("Clear All")
        self.Bclear.setToolTip('Reset image')
        self.Bclear.setFlat(True)
        self.Bclear.setStyleSheet('text-align:left')
        self.vlayout.addWidget(self.Bclear)    
        
        
        self.Btrain = QPushButton(self.centralWidget)
        self.Btrain.setObjectName("Btrain")
        self.Btrain.setText("Train Model")
        self.Btrain.setToolTip('Open training window')
        self.Btrain.setIcon(QIcon('icons/train.png'))
        self.Btrain.setFlat(True)
        self.Btrain.setStyleSheet('text-align:left')
        self.vlayout.addWidget(self.Btrain)    
            
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.predictlayout = QHBoxLayout(self.centralWidget)
        self.Bpredictall = QPushButton(self.centralWidget)
        self.Bpredictall.setObjectName("Bpredict")
        self.Bpredictall.setIcon(QIcon('icons/predict.png'))
        self.Bpredictall.setToolTip('Predict all images in prediction folder')
        self.Bpredictall.setText("Predict All")
        self.Bpredictall.setFlat(True)
        self.Bpredictall.setStyleSheet('text-align:left')
        self.Bpredict = QPushButton(self.centralWidget)
        self.Bpredict.setObjectName("Bpredict")
        self.Bpredict.setIcon(QIcon('icons/predict.png'))
        self.Bpredict.setToolTip('Predict current image')
        self.Bpredict.setText("Predict Current")
        self.Bpredict.setFlat(True)
        self.Bpredict.setStyleSheet('text-align:left')
        self.predictlayout.addWidget(self.Bpredictall)    
        self.predictlayout.addWidget(self.Bpredict)
        
        self.vlayout.addItem(self.predictlayout) 
        
        self.lslayout = QHBoxLayout(self.centralWidget)
        self.Bloadmodel = QPushButton(self.centralWidget)
        self.Bloadmodel.setObjectName("Bloadmodel")
        self.Bloadmodel.setIcon(QIcon('icons/loadmodel.png'))
        self.Bloadmodel.setToolTip('Load trained model')
        self.Bloadmodel.setText("Load")
        self.Bloadmodel.setFlat(True)
        self.Bloadmodel.setStyleSheet('text-align:left')
        
        self.Bsavemodel = QPushButton(self.centralWidget)
        self.Bsavemodel.setObjectName("Bsavemodel")
        self.Bsavemodel.setIcon(QIcon('icons/savemodel.png'))
        self.Bsavemodel.setToolTip('Save current model')
        self.Bsavemodel.setText("Save")
        self.Bsavemodel.setFlat(True)
        self.Bsavemodel.setStyleSheet('text-align:left')
        self.lslayout.addWidget(self.Bloadmodel)    
        self.lslayout.addWidget(self.Bsavemodel)
        
        self.vlayout.addItem(self.lslayout) 
        
        
        ##############
        self.classList = ClassList(self)
        self.classList.addClass('background')
        self.classList.addClass()
        self.vlayout.addWidget(self.classList)
        
        self.add_del_layout = QHBoxLayout(self.centralWidget)
        self.Baddclass = QPushButton(self.centralWidget)
        self.Baddclass.setText("+")
        self.Baddclass.setToolTip('Add a new class')
        self.Baddclass.setFlat(True)
        self.Baddclass.setMaximumHeight(bheight)
        self.add_del_layout.addWidget(self.Baddclass)
        self.Bdelclass = QPushButton(self.centralWidget)
        self.Bdelclass.setText("-")
        self.Bdelclass.setFlat(True)
        self.Bdelclass.setToolTip('Delete class')
        self.Bdelclass.setMaximumHeight(bheight)
        self.add_del_layout.addWidget(self.Bdelclass)
        self.vlayout.addItem(self.add_del_layout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.BPostProcessing = QPushButton(self.centralWidget)
        self.BPostProcessing.setText('Postprocessing')
        self.BPostProcessing.setStyleSheet('text-align:left')
        self.BPostProcessing.setObjectName("BPostProcessing")
        self.BPostProcessing.setIcon(QIcon('icons/postprocessing.png'))
        self.BPostProcessing.setToolTip('Open postprocessing window')
        self.BPostProcessing.setFlat(True)
        self.vlayout.addWidget(self.BPostProcessing)
        
        self.Bresults = QPushButton(self.centralWidget)
        self.Bresults.setText("Results")
        self.Bresults.setFlat(True)
        self.Bresults.setStyleSheet('text-align:left')
        self.Bresults.setIcon(QIcon('icons/results.png'))
        self.Bresults.setToolTip('Open results window')
        self.vlayout.addWidget(self.Bresults)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vlayout.addItem(verticalSpacer)
        
        self.nextprev_layout = QHBoxLayout(self.centralWidget)
        self.Bprev = QPushButton(self.centralWidget)
        self.Bprev.setObjectName("Bprev")
        self.Bprev.setText("<")
        self.Bprev.setFlat(True)
        self.Bprev.setToolTip('Previous window')
        self.nextprev_layout.addWidget(self.Bprev)
        self.Bnext = QPushButton(self.centralWidget)
        self.Bnext.setObjectName("Bnext")
        self.Bnext.setText(">")
        self.Bnext.setToolTip('Next window')
        self.Bnext.setFlat(True)
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
        

        
 
class ClassList(QListWidget):
    def __init__(self, parent):
        super(ClassList, self).__init__(parent.centralWidget)
        self.parent = parent
        self.setSelectionMode(QAbstractItemView.NoSelection)

    def addClass(self, name = 'class type'):
        item = QListWidgetItem(self)
        self.addItem(item)
        id = self.row(item)
        row = ClassWidget(name, id , self.parent)
        # not so nice to hard code size here
        item.setSizeHint(row.minimumSizeHint() + QSize(100,0) )
        self.setItemWidget(item, row)
        self.resize()
        self.parent.updateClassList()
        
    def removeClass(self):
        idx = self.count()-1
        if idx <= 1:
            return
        else:
            self.takeItem(idx)
            self.resize()
        if self.getActiveClass() == -1:
            self.setClass(0)    
        self.parent.updateClassList()

    def resize(self):
        dynamic_height = self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth()
        self.setFixedHeight(min(dynamic_height, 380))
        self.setFixedSize(self.sizeHintForColumn(0) + 2 * self.frameWidth(), self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth())
        
    def setClass(self, i):
        item = self.item(i)
        self.itemWidget(item).rb_select.setChecked(True)
        
    def getNumberOfClasses(self):
        return self.count()

    def getActiveClass(self):
        for i in range(self.count()):
            if self.itemWidget(self.item(i)).rb_select.isChecked():
                return i
        return -1
    
    def getClassColor(self, c):
        while c >= self.count() and c < 255:
            self.addClass()
        return self.itemWidget(self.item(c)).color
        

    
       
        
class ClassWidget(QWidget):
    ButtonGroup = QButtonGroup()
    def __init__(self, name, id, parent):
        super(ClassWidget, self).__init__(parent.centralWidget)
        self.id = id
        self.parent = parent
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        self.color = QColor(r,g,b)
        
        self.row = QHBoxLayout()
        self.rb_select = QRadioButton(parent)  
        self.rb_select.setToolTip('Select class')
        self.ButtonGroup.addButton(self.rb_select)

        self.pb_color = QPushButton()
        self.pb_color.setToolTip('Change color')
        self.pb_color.setMaximumSize(15, 15)
        self.setButtonColor(self.color)
        
        self.edit_name = QLineEdit(parent)
        self.edit_name.setText(name)
        self.edit_name.setToolTip('Edit class name')
        self.edit_name.setStyleSheet("QLineEdit { border: None }")
        
        self.label_numOfContours = QLabel(parent)
        self.label_numOfContours.setObjectName('numOfContours_class' + str(self.id))
        self.label_numOfContours.setToolTip('Number of contours')
        self.label_numOfContours.setText('0')
        
        
        self.row.addWidget(self.rb_select)
        self.row.addWidget(self.pb_color)
        self.row.addWidget(self.edit_name)
        self.row.addWidget(self.label_numOfContours)
        self.setLayout(self.row)
        
        self.pb_color.clicked.connect(self.changeColor)
        self.rb_select.toggled.connect(self.selectClass)
    
    def setButtonColor(self, color):
        self.color = color
        pixmap = QPixmap(15, 15)
        pixmap.fill(color)
        self.pb_color.setIcon(QIcon(pixmap))
        
    def changeColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setButtonColor(color)
            self.parent.classcolorChanged()

    def selectClass(self):
        self.parent.classChanged()
        
        
        