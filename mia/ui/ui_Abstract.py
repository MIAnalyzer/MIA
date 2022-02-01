# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:43:00 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm


class Window(object):
    def setupUi(self, Form):

        Form.setWindowTitle('Title') 
        styleForm(Form)
        
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

class TitleWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(TitleWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        