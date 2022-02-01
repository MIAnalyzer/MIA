# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
 
   
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from ui.style import styleForm, getHighlightColor, getBackgroundColor
from ui.ui_utils import DCDButton, saveFile
import math
import threading

class Window(object):
    def setupUi(self, Form):
        Form.setWindowTitle('Training Data') 
        styleForm(Form)
        self.centralWidget = QWidget(Form)
        layout = QVBoxLayout()
        self.centralWidget.setLayout(layout)

        self.figure = Figure(constrained_layout = True, dpi= 100)
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_size_inches(6, 6)

        self.BStop = DCDButton(self.centralWidget,'Stop Training')
        self.BStop.setIcon(QIcon('icons/train.png'))
        self.BSave = DCDButton(self.centralWidget)
        self.BSave.setIcon(QIcon('icons/savemodel.png'))
        self.BSave.setToolTip('Save training data')
        
        
        l1 = QHBoxLayout()

        self.LBatch = QLabel(self.centralWidget)
        self.LEpoch = QLabel(self.centralWidget)

        l1.addWidget(self.LBatch, stretch=99, alignment= Qt.AlignLeft)
        l1.addWidget(self.LEpoch, stretch=99, alignment= Qt.AlignRight)
        l1.addWidget(self.BSave, stretch=0)


        layout.addLayout(l1)
        layout.addWidget(self.canvas)  
        
        l2 = QHBoxLayout()
        l2.addWidget(self.BStop)

        
        layout.addLayout(l2)

        self.centralWidget.setFixedSize(layout.sizeHint())
        Form.setFixedSize(layout.sizeHint())

class TrainPlotWindow(QMainWindow, Window):
    def __init__(self,parent):
        super(TrainPlotWindow, self).__init__(parent)
        self.setupUi(self)
        self.parent = parent
        self.dl = parent.dl
        self.lock = threading.Lock()
        
        self.train_loss_axis = None
        self.train_metric_axis = None
        
        self.val_loss_axis = None
        self.val_metric_axis = None
        
        self.BStop.clicked.connect(self.Stop)
        self.BSave.clicked.connect(self.saveTrainingRecord)
        
    def hide(self):
        self.dl.interrupt()
        super(TrainPlotWindow, self).hide()
        
    def closeEvent(self, event):       
        self.dl.interrupt()
        super(TrainPlotWindow, self).closeEvent(event)
        
    def initialize(self):
        bg = getBackgroundColor()
        self.figure.patch.set_facecolor((bg.red()/255, bg.green()/255, bg.blue()/255))
        color = 'white'
        
        plt.rcParams['text.color'] = color
        plt.rcParams['axes.labelcolor'] = color
        plt.rcParams['axes.edgecolor'] = color
        plt.rcParams['xtick.color'] = color
        plt.rcParams['ytick.color'] = color

        
        if self.train_loss_axis is None or self.train_metric_axis is None: 
            gs = self.figure.add_gridspec(2, 2)
            self.train_loss_axis = self.figure.add_subplot(gs[0, 0])
            self.train_metric_axis = self.figure.add_subplot(gs[0, 1])
            self.train_loss_axis.set_facecolor((0.0, 0.0, 0.0))
            self.train_metric_axis.set_facecolor((0.0, 0.0, 0.0))
            
            
        if self.val_loss_axis is None or self.val_metric_axis is None: 
            self.val_loss_axis = self.figure.add_subplot(gs[1, 0])
            self.val_metric_axis = self.figure.add_subplot(gs[1, 1])
            self.val_loss_axis.set_facecolor((0.0, 0.0, 0.0))
            self.val_metric_axis.set_facecolor((0.0, 0.0, 0.0))

        # self.canvas.draw()
        self.refresh()

        
    def updatePlot(self):
        # to do make drawing faster, ie using blitting or animations
        # but be careful, there seems to be some memory leaks in matplotlib blitting
        # self.canvas.draw() is the time consuming part
        
        if self.lock.locked():
            return

        with self.lock:

            loss = self.dl.record.loss
            metric =  self.dl.record.metric
            val_loss = self.dl.record.val_loss
            val_metric = self.dl.record.val_metric
            lpe = self.dl.record.loss_per_epoch
            mpe = self.dl.record.metric_per_epoch

            
            self.train_loss_axis.clear()
            self.train_metric_axis.clear()

            
            if len(loss) > 0:
                self.train_loss_axis.set_ylabel('loss')
                self.train_loss_axis.set_xlabel('iteration')
    
            if len(self.dl.Model.metrics_names) > 1:
                self.train_metric_axis.set_ylabel(self.dl.Model.metrics_names[1])
                self.train_metric_axis.set_xlabel('iteration')
                
            if self.dl.data.validationData():
                self.val_loss_axis.set_visible(True)
                self.val_metric_axis.set_visible(True)
                self.val_loss_axis.clear()
                self.val_metric_axis.clear()
                self.val_loss_axis.set_ylabel('validation loss')
                self.val_loss_axis.set_xlabel('epoch')
               
                if len(self.dl.Model.metrics_names) > 1:
                    self.val_metric_axis.set_ylabel('validation ' + self.dl.Model.metrics_names[1])
                    self.val_metric_axis.set_xlabel('epoch')
            else:
                self.val_loss_axis.set_visible(False)
                self.val_metric_axis.set_visible(False)
                
            col = getHighlightColor()
            color = (col.red()/255, col.green()/255, col.blue()/255)

            self.train_loss_axis.plot(loss, '-' if len(loss) > 1 else '.', color = color)
            self.train_metric_axis.plot(metric, '-' if len(metric) > 1 else '.', color = color)  
            self.val_loss_axis.plot(val_loss, '-' if len(val_loss) > 1 else '.',  color='red' )
            self.val_loss_axis.plot(lpe, '--' if len(lpe) > 1 else '.', color=color )
            self.val_metric_axis.plot(val_metric, '-' if len(val_metric) > 1 else '.', color = 'red')
            self.val_metric_axis.plot(mpe, '--' if len(mpe) > 1 else '.', color = color)

    
            self.LBatch.setText("Batch: %d of %d" % (min(self.dl.data.getNumberOfBatches(),self.dl.record.currentbatch +1), self.dl.data.getNumberOfBatches()))
            self.LEpoch.setText("Epoch: %d of %d" % (min(self.dl.epochs,self.dl.record.currentepoch +1), self.dl.epochs))
            
            self.canvas.draw()


    def saveTrainingRecord(self):
        filename = saveFile("Save Results","Comma Separated File (*.csv)", 'csv') 
        if not filename:
            return
        try:
            self.dl.record.save(filename)  

        except PermissionError: 
            self.parent.emitPopup('Cannot write file (already open?)')
            return

          
    def refresh(self):
        
        thread = threading.Thread(target=self.updatePlot)
        thread.daemon = True
        thread.start()
        
        # self.updatePlot()

    def Stop(self):
        self.dl.interrupt()

    def Resume(self):
        self.parent.resumeTraining()

    def toggleTrainStatus(self, training):
        self.BStop.disconnect()
        if training:
            self.BStop.setText('Stop Training')
            self.BStop.setIcon(QIcon('icons/delete.png'))
            self.BStop.clicked.connect(self.Stop)
        else:
            self.BStop.setText('Resume Training')
            self.BStop.setIcon(QIcon('icons/train.png'))
            self.BStop.clicked.connect(self.Resume)
