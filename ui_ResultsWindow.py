# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:42:08 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import csv
import os
import glob
import Contour
import sys
from Tools import canvasTool

class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Results') 
        Form.setFixedSize(250, 80)
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(250, 80)

        self.vlayout = QVBoxLayout(self.centralWidget)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        self.CBSize = QCheckBox("Export scaled size",self.centralWidget)
        self.CBSize.setChecked(True)
        self.vlayout.addWidget(self.CBSize)
        
        self.hlayout = QHBoxLayout(self.centralWidget)
        self.BSetScale = QPushButton(self.centralWidget)
        self.BSetScale.setFlat(True)
        self.BSetScale.setIcon(QIcon('icons/measure.png'))
        self.BSetScale.setMaximumSize(28, 28)
        self.hlayout.addWidget(self.BSetScale)
        
        self.LEScale = QLineEdit(self.centralWidget)
        
        self.LEScale.setMaximumSize(100, 28)
        self.LEScale.setStyleSheet("border: None")
        self.LEScale.setStyleSheet('font:bold')
        self.LEScale.setAlignment(Qt.AlignRight)
        self.LEScale.setValidator(QIntValidator(1,999999999))
        self.hlayout.addWidget(self.LEScale)
        self.LScale = QLabel(self.centralWidget)
        self.LScale.setText('  pixel per mm')
        self.hlayout.addWidget(self.LScale)
        self.vlayout.addItem(self.hlayout)
        
        self.BExport = QPushButton(self.centralWidget)
        self.BExport.setText("Export as csv")
        self.BExport.setIcon(QIcon('icons/savemodel.png'))
        self.BExport.setFlat(True)
        self.vlayout.addWidget(self.BExport)
        


class ResultsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ResultsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.BExport.clicked.connect(self.saveResults)
        self.CBSize.clicked.connect(self.EnableSetScale)
        self.BSetScale.clicked.connect(self.setScaleTool)
        self.LEScale.textEdited.connect(self.setScale)
        self.LEScale.setText('1')
        
    def setScale(self, text):
        # qtvalidator does not have a status request so we need to validate again
        if self.LEScale.validator().validate(text,0)[0] == QValidator.Acceptable:
            self.parent.canvas.setScale(int(text))


    def setScaleTool(self):
        self.parent.canvas.setnewTool(canvasTool.scale.name)
        

    def EnableSetScale(self):
        if self.CBSize.isChecked():
            self.LEScale.setEnabled(True)
            self.BSetScale.setEnabled(True)
            self.LScale.setStyleSheet("color: black")
        else:
            self.LEScale.setEnabled(False)
            self.BSetScale.setEnabled(False)
            self.LScale.setStyleSheet("color: gray")

    def saveResults(self):
        filename = QFileDialog.getSaveFileName(self, "Save Results", '', "Comma Separated File (*.csv)")[0]
        if not filename.strip():
            return
        
        labels = glob.glob(os.path.join(self.parent.testImageLabelspath,'*.*'))
        if not labels:
            self.parent.PopupWarning('No predicted files')
            return
        labels = [x for x in labels if x.endswith(".npy") or x.endswith(".npz")]
        labels.sort()
        try:
            with open(filename, 'w', newline='') as csvfile:
                csvWriter = csv.writer(csvfile, delimiter = ';')
                header = ['image name'] + ['object number'] + ['object type'] + ['size']
                if self.CBSize.isChecked():
                    header += ['size in microns\u00b2']
                csvWriter.writerow(header)

                for i in labels:
                    contours = Contour.loadContours(i)
                    name = os.path.splitext(os.path.basename(i))[0]
                    
                    x = 0
                    for c in contours:
                        x += 1
                        row = [name] + [x] + [c.classlabel] + [c.getSize()]
                        if self.CBSize.isChecked():
                            row += ['%.2f'%(c.getSize()/((self.parent.canvas.scale_pixel_per_mm/1000)**2))]
                        csvWriter.writerow(row)
                    csvWriter.writerow([])
        except: 
            self.parent.PopupWarning('Cannot write file (already open?)')
            return
                    
        self.hide()
        

            

            
