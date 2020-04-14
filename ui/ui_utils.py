from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import random
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

class DCDButton(QPushButton):
    def __init__(self, parent, text=None):
        super(DCDButton, self).__init__(parent)
        self.parent = parent
        if text:
            self.setText(text)
            self.setStyleSheet('text-align:left')
        self.setFlat(True)


class ClassList(QListWidget):
    def __init__(self, parent):
        super(ClassList, self).__init__(parent)
        self.parent = parent
        self.setSelectionMode(QAbstractItemView.NoSelection)

    def addClass(self, name = 'class type'):
        item = QListWidgetItem(self)
        self.addItem(item)
        id = self.row(item)
        row = ClassWidget(name, id , self.parent)
        # not so nice to hard code size here
        item.setSizeHint(row.minimumSizeHint() + QSize(100,0) )
        self.setItemWidget(item, row)
        self.resize()
        self.parent.updateClassList()
        
    def removeClass(self):
        idx = self.count()-1
        if idx <= 1:
            return
        else:
            self.takeItem(idx)
            self.resize()
        if self.getActiveClass() == -1:
            self.setClass(0)    
        self.parent.updateClassList()

    def resize(self):
        dynamic_height = min(self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth(), 380)
        self.setFixedSize(self.sizeHintForColumn(0) + 2 * self.frameWidth(), dynamic_height)


    def setClass(self, i):
        item = self.item(i)
        self.itemWidget(item).rb_select.setChecked(True)
        
    def getNumberOfClasses(self):
        return self.count()

    def getActiveClass(self):
        for i in range(self.count()):
            if self.itemWidget(self.item(i)).rb_select.isChecked():
                return i
        return -1
    
    def getClassColor(self, c):
        while c >= self.count() and c < 255:
            self.addClass()
        return self.itemWidget(self.item(c)).color
        

        
class ClassWidget(QWidget):
    ButtonGroup = QButtonGroup()
    def __init__(self, name, id, parent):
        super(ClassWidget, self).__init__(parent)
        self.id = id
        self.parent = parent
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        self.color = QColor(r,g,b)
        
        self.row = QHBoxLayout()
        self.rb_select = QRadioButton(parent)  
        self.rb_select.setToolTip('Select class')
        self.ButtonGroup.addButton(self.rb_select)

        self.pb_color = QPushButton()
        self.pb_color.setToolTip('Change color')
        self.pb_color.setMaximumSize(15, 15)
        self.setButtonColor(self.color)
        
        self.edit_name = QLineEdit(parent)
        self.edit_name.setText(name)
        self.edit_name.setToolTip('Edit class name')
        self.edit_name.setStyleSheet("QLineEdit { border: None }")
        
        self.label_numOfContours = QLabel(parent)
        self.label_numOfContours.setObjectName('numOfContours_class' + str(self.id))
        self.label_numOfContours.setToolTip('Number of contours')
        self.label_numOfContours.setText('0')
        
        
        self.row.addWidget(self.rb_select)
        self.row.addWidget(self.pb_color)
        self.row.addWidget(self.edit_name)
        self.row.addWidget(self.label_numOfContours)
        self.setLayout(self.row)
        
        self.pb_color.clicked.connect(self.changeColor)
        self.rb_select.toggled.connect(self.selectClass)
    
    def setButtonColor(self, color):
        self.color = color
        pixmap = QPixmap(15, 15)
        pixmap.fill(color)
        self.pb_color.setIcon(QIcon(pixmap))
        
    def changeColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setButtonColor(color)
            self.parent.classcolorChanged()

    def selectClass(self):
        self.parent.classChanged()