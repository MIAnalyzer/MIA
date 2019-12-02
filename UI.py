# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:28:44 2019

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class MainWindow(object):
    def setupUi(self, Form):
        
        width = 1600 
        height = 1200

        
        Form.setWindowTitle('DeepCellDetector') 
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
        self.CBLearningMode.setObjectName("CBmode")
        self.vlayout.addWidget(self.CBLearningMode)
        
        self.Bload = QPushButton(self.centralWidget)
        self.Bload.setObjectName("Bload")
        self.Bload.setText("Set Folder")
        self.Bload.setFlat(True)
        self.vlayout.addWidget(self.Bload)
        
        
        bwidth = 28
        bheight = 28
        self.Buttonslayout = QHBoxLayout(self.centralWidget)
        self.Buttonslayout.setContentsMargins(0, 0, 0, 0)
        self.Bdrag = QPushButton(self.centralWidget)
        self.Bdrag.setMaximumSize(bwidth, bheight)
        self.Bdrag.setObjectName("Bdrag")
        self.Bdrag.setText("Drag")
        self.Bdrag.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdrag)

        self.Bdraw = QPushButton(self.centralWidget)
        self.Bdraw.setMaximumSize(bwidth, bheight)
        self.Bdraw.setObjectName("Bdraw")
        self.Bdraw.setText("Draw")
        self.Bdraw.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdraw)
        
        self.Bpoly = QPushButton(self.centralWidget)
        self.Bpoly.setMaximumSize(bwidth, bheight)
        self.Bpoly.setObjectName("Bpoly")
        self.Bpoly.setText("Poly")
        self.Bpoly.setFlat(True)
        self.Buttonslayout.addWidget(self.Bpoly)

        self.Bassign = QPushButton(self.centralWidget)
        self.Bassign.setMaximumSize(bwidth, bheight)
        self.Bassign.setObjectName("Bassign")
        self.Bassign.setText("Ass")
        self.Bassign.setFlat(True)
        self.Buttonslayout.addWidget(self.Bassign)
        
        self.Bextend = QPushButton(self.centralWidget)
        self.Bextend.setMaximumSize(bwidth, bheight)
        self.Bextend.setObjectName("Bexpand")
        self.Bextend.setText("Ext")
        self.Bextend.setFlat(True)
        self.Buttonslayout.addWidget(self.Bextend)
        
        
        self.Bdelete = QPushButton(self.centralWidget)
        self.Bdelete.setMaximumSize(bwidth, bheight)
        self.Bdelete.setObjectName("Bdelete")
        self.Bdelete.setText("X")
        self.Bdelete.setFlat(True)
        self.Buttonslayout.addWidget(self.Bdelete)


        self.vlayout.addItem(self.Buttonslayout)
        
        self.mode = QLabel(self.centralWidget)
        self.mode.setText("Drag")
        self.mode.setAlignment(Qt.AlignHCenter)
        self.vlayout.addWidget(self.mode)
        
        self.Bclear = QPushButton(self.centralWidget)
        self.Bclear.setObjectName("Bclear")
        self.Bclear.setText("Clear")
        self.Bclear.setFlat(True)
        self.vlayout.addWidget(self.Bclear)
        
        self.classeslayouts = []
        self.RBClasses = []
        self.ColorClasses = []
        
        
        for i in range(11):
            layout = QHBoxLayout(self.centralWidget)
            layout.setSpacing(5)
            button = QRadioButton(self.centralWidget)
            button.setObjectName("class" + str(i))
            button.classNum = i
            self.RBClasses.append(button)
            layout.addWidget(button)  
            color = QLabel(self.centralWidget)
            color.setText("    ")
            color.setStyleSheet("background-color: rgb(255, 0, 0)")
            layout.addWidget(color)
            self.ColorClasses.append(color) 
            edit = QLineEdit(self.centralWidget)
            if i == 0:
                edit.setText("Background")
                edit.setEnabled(False)
            else:
                edit.setText("Class" + " " + str(i))
            edit.setStyleSheet("QLineEdit { border: None }")
            layout.addWidget(edit)
            self.classeslayouts.append(layout)
            self.vlayout.addItem(layout)
                
        self.Btrain = QPushButton(self.centralWidget)
        self.Btrain.setObjectName("Btrain")
        self.Btrain.setText("Train")
        self.Btrain.setFlat(True)
        self.vlayout.addWidget(self.Btrain)    
            
        
        self.Bpredict = QPushButton(self.centralWidget)
        self.Bpredict.setObjectName("Bpredict")
        self.Bpredict.setText("Predict")
        self.Bpredict.setFlat(True)
        self.vlayout.addWidget(self.Bpredict) 
        
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
        Form.setCentralWidget(self.centralWidget)