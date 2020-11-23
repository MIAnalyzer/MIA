# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:06:36 2020

@author: Koerber
"""
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import json
from ui.ui_utils import QSetting
from ui.ui_utils import ClassList

class Settings: 
    def __init__(self,parent):
        self.parent = parent
        self.settingsDict = {}
    
    
    def saveSettings(self, path):
        self.collectSettings()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.settingsDict, f, ensure_ascii=False, indent=4)
        except:
            # can not save
            pass

        
    def loadSettings(self, path):
        self.settingsDict.clear()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.settingsDict = json.load(f)
            self.applySettings()
            self.parent.settingsLoaded()
        except:
            # can not load
            pass

            
    
    def collectSettings(self):
        self.settingsDict.clear()
        self.addObjectSetting(QCheckBox, QCheckBox.isChecked)
        self.addObjectSetting(QRadioButton, QRadioButton.isChecked)
        self.addObjectSetting(QSlider, QSlider.value)
        self.addObjectSetting(QComboBox, QComboBox.currentIndex)
        self.addObjectSetting(QDoubleSpinBox, QDoubleSpinBox.value)
        self.addObjectSetting(QSpinBox, QSpinBox.value)
        self.addObjectSetting(QGroupBox, QGroupBox.isChecked)
        self.addObjectSetting(QLineEdit, QLineEdit.text)
        self.addObjectSetting(QSetting, QSetting.value)

   
    def addObjectSetting(self, objectType, function):
        items = self.parent.findChildren(objectType)
        for i in items:
            if i.objectName() != '' and i.objectName() != 'qt_spinbox_lineedit': # spinbox line edit has qt_spinbox_lineedit as default name
                self.settingsDict[i.objectName()] = function(i)
        
            
    def applySettings(self):
        self.parent.setNumberOfClasses(self.settingsDict['NumOfClasses'])
        for obj in self.settingsDict:
            widget = self.parent.findChild(QWidget, obj)
            if isinstance(widget, QCheckBox):
                widget.setChecked(self.settingsDict[obj])
            elif isinstance(widget, QRadioButton):
                widget.setChecked(self.settingsDict[obj])
            elif isinstance(widget, QSlider):
                widget.setValue(self.settingsDict[obj])
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(self.settingsDict[obj])
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(self.settingsDict[obj])
            elif isinstance(widget, QSpinBox):
                widget.setValue(self.settingsDict[obj])
            elif isinstance(widget, QGroupBox):
                widget.setChecked(self.settingsDict[obj])  
            elif isinstance(widget, QLineEdit):
                widget.setText(self.settingsDict[obj])  
            elif isinstance(widget, QSetting):
                widget.setValue(self.settingsDict[obj])  
            # else:
                # do nothing

        