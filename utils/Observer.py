# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:37:31 2020

@author: Koerber
"""

from PyQt5.QtCore import *
from abc import ABC, abstractmethod
import traceback
import sys


class Observer(ABC):

    @abstractmethod
    def Error(self, exctype, value, traceback):
        pass
    @abstractmethod
    def Finished(self):
        pass
    @abstractmethod
    def Result(self,result):
        pass
    @abstractmethod
    def Progress(self,i):
        pass
    @abstractmethod
    def Started(self):
        pass



class InterfaceSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal()
    start = pyqtSignal()
    epoch_end = pyqtSignal(int)

class dlObservable():
    def __init__(self):
        self.observers = []
        
    def attachObserver(self,o):
        self.observers.append(o)
        
    def detachObserver(self,o):
        self.observers.remove(o)
        
    def notifyError(self):
        for o in self.observers:
            exctype, value, tb = sys.exc_info()
            o.Error(exctype, value, tb)
            
    def notifyPredictionFinished(self, prediction):
        for o in self.observers:
            o.Result(prediction)            

    def notifyTrainingStarted(self):
        for o in self.observers:
            o.Started()            

    def notifyTrainingFinished(self):
        for o in self.observers:
            o.Finished()
        
    def notifyTrainingResult(self, result):
        for o in self.observers:
            o.Result(result)
        
    def notifyTrainingProgress(self):
        for o in self.observers:
            o.Progress()
            
    def notifyEpochEnd(self, epoch):
        for o in self.observers:
            o.EpochEnd(epoch)
    

class QtObserver(Observer):
    def __init__(self):
        self.signals = InterfaceSignals()  
        
    def Error(self,exctype, value, traceback):
        self.signals.error.emit((exctype, value, traceback))
        
    def Finished(self):
        self.signals.finished.emit()
        
    def Result(self, result):
        self.signals.result.emit(result)
        
    def Progress(self):
        self.signals.progress.emit()
        
    def Started(self):
        self.signals.start.emit()
        
    def EpochEnd(self, epoch):
        self.signals.epoch_end.emit(epoch)
        
    
        