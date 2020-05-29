# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


        
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ui.style import styleForm, getHighlightColor, getBackgroundColor
import random

class Window(object):
    def setupUi(self, Form):
        width = 600
        height= 300
        Form.setWindowTitle('Training Data') 
        Form.setFixedSize(width, height)
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        layout = QVBoxLayout()
        self.centralWidget.setLayout(layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.button = QPushButton('Stop Training')

        layout.addWidget(self.canvas)
        layout.addWidget(self.button)



class TrainPlotWindow(QMainWindow, Window):
    def __init__(self,parent):
        super(TrainPlotWindow, self).__init__()
        self.setupUi(self)
        self.parent = parent
        
        self.ax1 = None
        self.ax2 = None
        
        self.button.clicked.connect(self.Stop)
        
    def initialize(self):
        # to do set real colors
        bg = getBackgroundColor()
        self.figure.patch.set_facecolor((bg.red()/255, bg.green()/255, bg.blue()/255))
        col = getHighlightColor()
        # COLOR = (col.red()/255, col.green()/255, col.blue()/255)
        color = 'white'
        
        plt.rcParams['text.color'] = color
        plt.rcParams['axes.labelcolor'] = color
        plt.rcParams['axes.edgecolor'] = color
        plt.rcParams['xtick.color'] = color
        plt.rcParams['ytick.color'] = color
        
        
        gs = self.figure.add_gridspec(1, 2)
        self.ax1 = self.figure.add_subplot(gs[0, 0])
        self.ax2 = self.figure.add_subplot(gs[0, 1])
        self.ax1.set_facecolor((0.0, 0.0, 0.0))
        self.ax2.set_facecolor((0.0, 0.0, 0.0))
        self.ax1.clear()
        self.ax2.clear()
        
        # self.ax1.set_title('loss')
        # self.ax1.set_xlabel('iteration')
        
        # self.ax2.set_title(self.parent.parent.Model.metrics_names[1])
        # self.ax2.set_xlabel('iteration')
        

        # self.ax1.plot([])
        # self.ax2.plot([])

        self.canvas.draw()
        self.refresh()

        
    def updatePlot(self, loss ,metric):

        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.set_title('loss')
        self.ax1.set_xlabel('iteration')
        
        self.ax2.set_title(self.parent.parent.Model.metrics_names[1])
        self.ax2.set_xlabel('iteration')

        col = getHighlightColor()
        color = (col.red()/255, col.green()/255, col.blue()/255)
        self.ax1.plot(loss, '-', color = color)
        self.ax2.plot(metric, '-', color = color)

        self.canvas.draw()
          
    def refresh(self):
        self.updatePlot(self.parent.loss, self.parent.metric)
        QApplication.processEvents()

    def Stop(self):
        print(self.parent)
        self.parent.interrupt = True