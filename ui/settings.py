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
    
    
    def saveSettings(self, path, networksettingsonly=False):
        self.collectSettings(networksettingsonly)
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

            
    
    def collectSettings(self, nnsettingsonly):
        self.settingsDict.clear()
        self.addObjectSetting(QCheckBox, QCheckBox.isChecked, nnsettingsonly)
        self.addObjectSetting(QRadioButton, QRadioButton.isChecked, nnsettingsonly)
        self.addObjectSetting(QSlider, QSlider.value, nnsettingsonly)
        self.addObjectSetting(QComboBox, QComboBox.currentIndex, nnsettingsonly)
        self.addObjectSetting(QDoubleSpinBox, QDoubleSpinBox.value, nnsettingsonly)
        self.addObjectSetting(QSpinBox, QSpinBox.value, nnsettingsonly)
        self.addObjectSetting(QGroupBox, QGroupBox.isChecked, nnsettingsonly)
        self.addObjectSetting(QLineEdit, QLineEdit.text, nnsettingsonly)
        self.addObjectSetting(QSetting, QSetting.value, nnsettingsonly)

   
    def addObjectSetting(self, objectType, function, nnsettingsonly):
        items = self.parent.findChildren(objectType)
        for i in items:
            if i.objectName() != '' and i.objectName() != 'qt_spinbox_lineedit': # spinbox line edit has qt_spinbox_lineedit as default name
                if not nnsettingsonly:
                    self.settingsDict[i.objectName()] = function(i)
                else:
                    if i.objectName().startswith('nn'):
                        self.settingsDict[i.objectName()] = function(i)
        
            
    def applySettings(self):
        self.parent.setNumberOfClasses(self.settingsDict['nn_NumOfClasses'])
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

        