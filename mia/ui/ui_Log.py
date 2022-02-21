# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:38:21 2022

@author: Koerber
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from ui.ui_utils import ScrollLabel

class Window(object):
    def setupUi(self, Form):

        Form.setWindowTitle('Log File') 
        styleForm(Form)
        
        self.LogText = ScrollLabel(Form)
        self.setCentralWidget(self.LogText)
        
        self.resize(650,350)
        

class LogWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(LogWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)

    def showLog(self, path):
        f = open(path, "r")
        self.LogText.setText(f.read())
