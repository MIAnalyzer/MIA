# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:38:21 2022

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm


class Window(object):
    def setupUi(self, Form):

        Form.setWindowTitle('Log File') 
        styleForm(Form)
        
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)

        self.Text = QLabel(self.centralWidget)
        self.Text .setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.Text.setWordWrap(True)
        self.Scroll = QScrollArea(self.centralWidget)
        self.Scroll.setWidget(self.Text)
        self.Text.setFixedWidth(500)
        self.Text.setFixedHeight(250)

        
        self.vlayout.addWidget(self.Scroll)
        
        # self.vlayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

class LogWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(LogWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        
        
    def showLog(self, path):
        f = open(path, "r")
        self.Text.setText(f.read())
        # self.Text.resize(self.Text.sizeHint())