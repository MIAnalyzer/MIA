from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import random

from ui.style import getBrightColor

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
        self.SpinBox.lineEdit().setObjectName('')
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)
        
    def objectName(self):
        return self.SpinBox.objectName(name)
        
    def setObjectName(self, name):
        self.SpinBox.setObjectName(name)

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)
        

class LabelledDoubleSpinBox(QWidget):
    def __init__(self,labeltext, parent):
        super(LabelledDoubleSpinBox, self).__init__(parent)

        
        layout = QHBoxLayout()
        self.SpinBox = QDoubleSpinBox(parent)
        self.SpinBox.lineEdit().setObjectName('')
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)
        
    def objectName(self):
        return self.SpinBox.objectName(name)
        
    def setObjectName(self, name):
        self.SpinBox.setObjectName(name)

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)

class LabelledAdaptiveDoubleSpinBox(QWidget):
    def __init__(self,labeltext, parent):
        super(LabelledAdaptiveDoubleSpinBox, self).__init__(parent)
        
        layout = QHBoxLayout()
        self.SpinBox = QAdaptiveDoubleSpinBox(parent)
        self.SpinBox.lineEdit().setObjectName('')
        layout.addWidget(self.SpinBox)
        self.Label = QLabel(parent)
        self.Label.setText(labeltext)
        layout.addWidget(self.Label)
        self.setLayout(layout)
        
    def objectName(self):
        return self.SpinBox.objectName(name)
        
    def setObjectName(self, name):
        self.SpinBox.setObjectName(name)
        

    def setToolTip(self, tooltip):
        self.Label.setToolTip(tooltip)
        self.SpinBox.setToolTip(tooltip)

class DCDButton(QPushButton):
    isclicked = pyqtSignal()
    
    def __init__(self, parent, text=None):
        super(DCDButton, self).__init__(parent)
        self.parent = parent
        if text:
            self.setText(text)
            self.setStyleSheet('text-align:left')
        self.setStyleSheet('QPushButton:hover:!pressed { border: 1px solid lightblue}')
        self.setFlat(True)
        self.clicked.connect(self.on_clicked)
        
    def on_clicked(self):
        self.isclicked.emit()
        self.parent.setFocus()
        
        
    def resetStyle(self):
        self.setStyleSheet('QPushButton:hover:!pressed { border: 1px solid lightblue}')
        
        
        
class ScrollLabel(QScrollArea):
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)
 
        self.setWidgetResizable(True)
        content = QWidget(self)
        self.setWidget(content)
        layout = QVBoxLayout(content)
        self.label = QLabel(content)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
 
    def setText(self, text):
        self.label.setText(text)
        
        
class QSetting(QWidget):
    valueChanged = pyqtSignal() 
    def __init__(self, parent, name, value=None):
        super(QWidget, self).__init__(parent)
        self._value = value
        self.setObjectName(name)
        self.setVisible(False)
        
    def value(self):
        return self._value
    
    def setValue(self, v):
        self._value = v
        self.valueChanged.emit()


class ClassList(QListWidget):
    def __init__(self, parent):
        super(ClassList, self).__init__(parent)
        self.parent = parent
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setStyleSheet("QListWidget { border: 2px solid " + getBrightColor(asString=True) +";}")

    def addClass(self, name = 'class type'):
        item = QListWidgetItem(self)
        self.addItem(item)
        id = self.row(item)
        row = ClassWidget(name, id , self.parent)
        # not so nice to hard code size here
        item.setSizeHint(row.minimumSizeHint() + QSize(100,0) )
        self.setItemWidget(item, row)
        self.resize()
        self.parent.ClassListUpdated()
        
    def removeClass(self):
        idx = self.count()-1
        if idx <= 1:
            return
        else:
            self.takeItem(idx)
            self.resize()
        if self.getActiveClass() == -1:
            self.setClass(0)    
        self.parent.ClassListUpdated()

    def resize(self):
        dynamic_height = min(self.sizeHintForRow(0) * self.count() + 2 * self.frameWidth(), 380)
        self.setFixedSize(self.sizeHintForColumn(0) + 2 * self.frameWidth(), dynamic_height)

    def setClass(self, i):
        item = self.item(i)
        if item:
            self.itemWidget(item).rb_select.setChecked(True)
            
    def setClassWeight(self, classnum, weight):
        item = self.item(classnum)
        if item:
            self.itemWidget(item).classweight.setValue(weight)
            
    def getClassWeight(self, classnum):
        item = self.item(classnum)
        if item:
            return self.itemWidget(item).classweight.value()
            
    def getClassName(self, i):
        item = self.item(i)
        if item:
            return self.itemWidget(item).edit_name.text()
        
    def getNumberOfClasses(self):
        return self.count()
    
    def setNumberOfClasses(self, numclasses):
        diff = numclasses - self.getNumberOfClasses()
        if diff > 0:
            for i in range(diff):
                self.addClass()
        if diff < 0:
            for i in range(abs(diff)):
                self.removeClass()
            

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

        self.pb_color = DCDButton(parent)
        self.pb_color.setToolTip('Change color')
        self.pb_color.setMaximumSize(15, 15)
        self.setButtonColor(self.color)
        
        self.edit_name = QLineEdit(parent)
        self.edit_name.setText(name)
        self.edit_name.setToolTip('Edit class name')
        self.edit_name.setObjectName('nn_class_%i' % self.id)
        self.edit_name.setStyleSheet("QLineEdit { border: None }")

        self.classweight = QSetting(parent, 'nn_classweight_%i' % self.id, 0.5)
        
        
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
        self.edit_name.textChanged.connect(self.nameChanged)

    
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
        
    def nameChanged(self):
        self.parent.ClassListUpdated()


# open dialogs
def openFolder(header):
    dialog = QFileDialog()
    dialog.setWindowTitle(header)
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.ShowDirsOnly, False)
    
    if dialog.exec() == QDialog.Accepted:
        return dialog.selectedFiles()[0]
    else:
        return False

def saveFile(header, filter, suffix, defaultname = None):
    dialog = QFileDialog()
    dialog.setWindowTitle(header)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setDefaultSuffix(suffix)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    if defaultname:
        dialog.selectFile(defaultname)
    dialog.setNameFilters([filter])
    if dialog.exec_() == QDialog.Accepted:
        return dialog.selectedFiles()[0]
    else:
       return False

def loadFile(header, filter):
    dialog = QFileDialog()
    dialog.setWindowTitle(header)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setNameFilters([filter])
    if dialog.exec_() == QDialog.Accepted:
        return dialog.selectedFiles()[0]
    else:
       return False