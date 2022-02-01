# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:42:08 2020

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from utils.shapes.ImageLabel import loadImageLabel


class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Classification Settings') 

        styleForm(Form)
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(6)
        self.centralWidget.setLayout(self.vlayout)
        
        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

class ClassificationResultsWindow(QMainWindow, Window):
    def __init__(self, parent, parentwindow):
        super(ClassificationResultsWindow, self).__init__(parent)
        self.parent = parent
        self.parentwindow = parentwindow
        self.setupUi(self)
        self.funcs = []


    def saveClasses(self, writer, images, labels):
        header = ['image name'] + ['frame number'] + ['class'] + ['classname']
        writer.writerow(header)
        num = len(labels)
        self.parent.emitinitProgress(num)
    
        for image, label in zip(images,labels):
            name = self.parent.files.getFilenameFromPath(self.parent.files.convertIfStackPath(image),withfileextension=True)
            classlabel,_ = loadImageLabel(self.parent.files.convertIfStackPath(label))
            if self.parent.files.isStackLabel(label):
                frame = self.parent.files.getFrameNumber(label) + 1
            else:
                frame = 1
            classname = self.parent.classList.getClassName(classlabel.classlabel)
            row = [name] + [frame] + [classlabel.classlabel] + [classname]
            writer.writerow(row) 
    