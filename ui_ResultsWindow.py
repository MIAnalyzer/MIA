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
        
        self.CBSize = QCheckBox("Export Size")
        self.CBSize.setChecked(True)
        self.CBNumber= QCheckBox("Export Number")
        self.CBNumber.setChecked(True)
        self.vlayout.addWidget(self.CBSize)
        self.vlayout.addWidget(self.CBNumber)
        
        self.BExport = QPushButton(self.centralWidget)
        self.BExport.setText("Export as xls")
        self.BExport.setIcon(QIcon('icons/savemodel.png'))
        self.BExport.setFlat(True)
        self.vlayout.addWidget(self.BExport)
        


class ResultsWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ResultsWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.BExport.clicked.connect(self.saveResults)


    def saveResults(self):
        if self.parent.testImageLabelspath is None:
            self.parent.PopupWarning('Prediction folder not selected')
            return
        
        self.parent.canvas.createLabelFromContours()
        filename = QFileDialog.getSaveFileName(self, "Save Results", '', "Comma Separated File (*.csv)")[0]
        if not filename.strip():
            return
        
        labels = glob.glob(os.path.join(self.parent.testImageLabelspath,'*.*'))
        if not labels:
            self.parent.PopupWarning('No predicted files')
            return
        labels = [x for x in labels if x.endswith(".npy") or x.endswith(".npz")]
        labels.sort()
        savesize = self.CBSize.isChecked()
        savenumber = self.CBNumber.isChecked()
        try:
            with open(filename, 'w', newline='') as csvfile:
                csvWriter = csv.writer(csvfile, delimiter = ';')
                csvWriter.writerow(['number']+['type'] + ['size'])
                for i in labels:
                    contours = Contour.loadContours(i)
                    name = os.path.splitext(os.path.basename(i))[0]
                    csvWriter.writerow([name])
                    x = 0
                    for c in contours:
                        x += 1
                        csvWriter.writerow([x] + [c.classlabel] + [c.getSize()])
                    csvWriter.writerow([])
        except: 
            self.parent.PopupWarning('Cannot write file (already open?)')
            return
                    
        self.hide()
        

            

            
