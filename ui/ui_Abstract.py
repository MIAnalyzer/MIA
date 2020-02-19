# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Window(object):
    def setupUi(self, Form):
        width = 250
        height= 80
        Form.setWindowTitle('Title') 
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)

class TitleWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TitleWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        