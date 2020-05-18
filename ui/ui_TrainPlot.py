# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


        
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import random

class Window(object):
    def setupUi(self, Form):
        width = 450
        height= 300
        Form.setWindowTitle('Training Data') 
        Form.setFixedSize(width, height)
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        layout = QVBoxLayout()
        self.centralWidget.setLayout(layout)

        # a figure instance to plot on
        self.figure = Figure()
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        self.button = QPushButton('Stop Training')

        layout.addWidget(self.canvas)
        layout.addWidget(self.button)



class TrainPlotWindow(QMainWindow, Window):
    def __init__(self,parent):
        super(TrainPlotWindow, self).__init__()
        self.setupUi(self)
        self.parent = parent
        self.button.clicked.connect(self.Stop)

    def plot(self, data):
        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        ax.clear()
        # plot data
        ax.plot(data, '*-')
        # refresh canvas
        self.canvas.draw()
          
    def refresh(self):
        QApplication.processEvents()

    def Stop(self):
        print(self.parent)
        self.parent.interrupt = True