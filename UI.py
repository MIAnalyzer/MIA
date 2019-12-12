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
        
        width = 1600 
        height = 1200
        

        
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
        
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.hlayout.setContentsMargins(0, 0, 0, 0)
        self.hlayout.setSpacing(6)
        
        
        self.vlayout = QVBoxLayout()
        verticalSpacer = QSpacerItem(20, 40)
        self.vlayout.addItem(verticalSpacer)
        
        ## vertical layout - buttons
        self.CBLearningMode = QComboBox(self.centralWidget)
        self.CBLearningMode.addItem("Classification")
        self.CBLearningMode.addItem("Segmentation")
        self.CBLearningMode.addItem("Time Series")
        self.CBLearningMode.setObjectName("CBmode")
        self.CBLearningMode.setEditable(False)
        self.CBLearningMode.setFocus(False)
        self.vlayout.addWidget(self.CBLearningMode)
        
        
        bwidth = 28
        bheight = 28
        
        self.TrainImagelayout = QHBoxLayout(self.centralWidget)
        self.BLoadTrainImages = QPushButton(self.centralWidget)
        self.BLoadTrainImages.setObjectName("BLoadTrainImages")
        self.BLoadTrainImages.setMaximumSize(bwidth, bheight)
        self.BLoadTrainImages.setIcon(QIcon('icons/load.png'))
        self.BLoadTrainImages.setFlat(True)
        self.BTrainImageFolder = QPushButton(self.centralWidget)
        self.BTrainImageFolder.setText("Set Training Image Folder")
        self.BTrainImageFolder.setObjectName("BTrainImageFolder")
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
        self.BLoadTestImages.setIcon(QIcon('icons/load.png'))
        self.BLoadTestImages.setFlat(True)
        self.BTestImageFolder = QPushButton(self.centralWidget)
        self.BTestImageFolder.setText("Set Training Image Folder")
        self.BTestImageFolder.setObjectName("BTestImageFolder")
        self.BTestImageFolder.setFlat(True)
        self.BTestImageFolder.setStyleSheet('text-align:left')
        self.BTestImageFolder.setEnabled(False)
        self.TestImagelayout.addWidget(self.BLoadTestImages)
        self.TestImagelayout.addWidget(self.BTestImageFolder)
        self.vlayout.addItem(self.TestImagelayout)
        
        
        
        
        self.Buttonslayout = QHBoxLayout(self.centralWidget)
        self.Buttonslayout.setContentsMargins(0, 0, 0, 0)
        self.Bdrag = QPushButton(self.centralWidget)
        self.Bdrag.setMaximumSize(bwidth, bheight)
        self.Bdrag.setObjectName("Bdrag")
        self.Bdrag.setIcon(QIcon('icons/drag.png'))
        self.Bdrag.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdrag)

        self.Bdraw = QPushButton(self.centralWidget)
        self.Bdraw.setMaximumSize(bwidth, bheight)
        self.Bdraw.setObjectName("Bdraw")
        self.Bdraw.setIcon(QIcon('icons/draw.png'))
        self.Bdraw.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdraw)
        
        self.Bpoly = QPushButton(self.centralWidget)
        self.Bpoly.setMaximumSize(bwidth, bheight)
        self.Bpoly.setObjectName("Bpoly")
        self.Bpoly.setIcon(QIcon('icons/poly.png'))
        self.Bpoly.setFlat(True)
        self.Buttonslayout.addWidget(self.Bpoly)

        self.Bassign = QPushButton(self.centralWidget)
        self.Bassign.setMaximumSize(bwidth, bheight)
        self.Bassign.setObjectName("Bassign")
        self.Bassign.setIcon(QIcon('icons/assign.png'))
        self.Bassign.setFlat(True)
        self.Buttonslayout.addWidget(self.Bassign)
        
        self.Bextend = QPushButton(self.centralWidget)
        self.Bextend.setMaximumSize(bwidth, bheight)
        self.Bextend.setObjectName("Bexpand")
        self.Bextend.setIcon(QIcon('icons/expand.png'))
        self.Bextend.setFlat(True)
        self.Buttonslayout.addWidget(self.Bextend)
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)
        
        self.Bdelete = QPushButton(self.centralWidget)
        self.Bdelete.setMaximumSize(bwidth, bheight)
        self.Bdelete.setObjectName("Bdelete")
        self.Bdelete.setIcon(QIcon('icons/delete.png'))
        self.Bdelete.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdelete)

        self.vlayout.addItem(self.Buttonslayout)
        
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)

            
        self.Bclear = QPushButton(self.centralWidget)
        self.Bclear.setObjectName("Bclear")
        self.Bclear.setIcon(QIcon('icons/clear.png'))
        self.Bclear.setText("Clear")
        self.Bclear.setFlat(True)
        self.Bclear.setStyleSheet('text-align:left')
        self.vlayout.addWidget(self.Bclear)    
        
        self.Btrain = QPushButton(self.centralWidget)
        self.Btrain.setObjectName("Btrain")
        self.Btrain.setText("Train")
        self.Btrain.setIcon(QIcon('icons/train.png'))
        self.Btrain.setFlat(True)
        self.Btrain.setStyleSheet('text-align:left')
        self.vlayout.addWidget(self.Btrain)    
            
        
        self.Bpredict = QPushButton(self.centralWidget)
        self.Bpredict.setObjectName("Bpredict")
        self.Bpredict.setIcon(QIcon('icons/predict.png'))
        self.Bpredict.setText("Predict")
        self.Bpredict.setFlat(True)
        self.Bpredict.setStyleSheet('text-align:left')
        self.vlayout.addWidget(self.Bpredict) 
        
      
        
        
        ##############
        self.classList = ClassList(self)
        self.classList.addClass('background')
        self.classList.addClass()
        self.vlayout.addWidget(self.classList)
        
        self.add_del_layout = QHBoxLayout(self.centralWidget)
        self.Baddclass = QPushButton(self.centralWidget)
        self.Baddclass.setText("+")
        self.Baddclass.setFlat(True)
        self.Baddclass.setMaximumHeight(bheight)
        self.add_del_layout.addWidget(self.Baddclass)
        self.Bdelclass = QPushButton(self.centralWidget)
        self.Bdelclass.setText("-")
        self.Bdelclass.setFlat(True)
        self.Bdelclass.setMaximumHeight(bheight)
        self.add_del_layout.addWidget(self.Bdelclass)
        self.vlayout.addItem(self.add_del_layout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.vlayout.addWidget(line)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vlayout.addItem(verticalSpacer)
        
        self.nextprev_layout = QHBoxLayout(self.centralWidget)
        self.Bprev = QPushButton(self.centralWidget)
        self.Bprev.setObjectName("Bprev")
        self.Bprev.setText("<")
        self.Bprev.setFlat(True)
        self.nextprev_layout.addWidget(self.Bprev)
        self.Bnext = QPushButton(self.centralWidget)
        self.Bnext.setObjectName("Bnext")
        self.Bnext.setText(">")
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
        
        self.StatusMode = QLabel(self.centralWidget)
        self.StatusMode.setText('Current Mode: Drag')
        self.StatusMode.setObjectName("StatusMode")
        self.statusBar().addPermanentWidget(self.StatusMode, stretch = 0)
        
 
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
        item.setSizeHint(row.minimumSizeHint())
        self.setItemWidget(item, row)
        self.resize()
        
    def removeClass(self):
        idx = self.count()-1
        if idx <= 1:
            return
        else:
            self.takeItem(idx)
            self.resize()
        if self.getActiveClass() == -1:
            self.SetClass(0)    

    def resize(self):
        dynamic_height = self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth()
        self.setFixedHeight(min(dynamic_height, 380))
        #self.setFixedSize(self.sizeHintForColumn(0) + 2 * self.frameWidth(), self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth())
        
    def SetClass(self, i):
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
        self.ButtonGroup.addButton(self.rb_select)

        self.pb_color = QPushButton()
        self.pb_color.setMaximumSize(15, 15)
        self.setButtonColor(self.color)
        
        self.edit_name = QLineEdit(parent)
        self.edit_name.setText(name)
        self.edit_name.setStyleSheet("QLineEdit { border: None }")
        self.row.addWidget(self.rb_select)
        self.row.addWidget(self.pb_color)
        self.row.addWidget(self.edit_name)
        self.setLayout(self.row)
        
        self.pb_color.clicked.connect(self.changeColor)
    
    def setButtonColor(self, color):
        self.color = color
        pixmap = QPixmap(15, 15)
        pixmap.fill(color)
        self.pb_color.setIcon(QIcon(pixmap))
        
    def changeColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setButtonColor(color)
            self.parent.colorChanged()

        
        
        