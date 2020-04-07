from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class QAdaptiveDoubleSpinBox(QDoubleSpinBox):
    # implementation of QDoubleSpinBox with adaptive step size,
    # will be obsolete in future as QAbstractSpinBox::AdaptiveDecimalStepType is supported with qt 9.12 (not in python for now)
    def __init__(self, parent):
        super(QAdaptiveDoubleSpinBox, self).__init__(parent)
        self.epsilon = 1e-8
        
    def stepBy(self,steps):
        value = self.value()
        
        if steps < 0:
            self.setSingleStep(value/2)
        else:
            self.setSingleStep(value)

        QDoubleSpinBox.stepBy(self,steps)
        
    def textFromValue(self, value):
        return QLocale().toString(round(value, 6), 'f', QLocale.FloatingPointShortest)

class LabelledSpinBox(QWidget):
    def __init__(self,labeltext, parent):
        super(LabelledSpinBox, self).__init__(parent)

        layout = QHBoxLayout()
        self.SpinBox = QSpinBox(parent)
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)

class LabelledDoubleSpinBox(QWidget):
    def __init__(self,labeltext, parent):
        super(LabelledDoubleSpinBox, self).__init__(parent)

        layout = QHBoxLayout()
        self.SpinBox = QDoubleSpinBox(parent)
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)

class LabelledAdaptiveDoubleSpinBox(QWidget):
    def __init__(self,labeltext, parent):
        super(LabelledAdaptiveDoubleSpinBox, self).__init__(parent)

        layout = QHBoxLayout()
        self.SpinBox = QAdaptiveDoubleSpinBox(parent)
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)

