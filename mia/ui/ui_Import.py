

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ui.style import styleForm
from ui.ui_utils import DCDButton, openFolder

from utils.shapes.Contour  import extractContoursFromLabel, saveContours
from dl.method.mode import dlMode
import cv2
import glob
import os


class Window(object):
    def setupUi(self, Form):

        Form.setWindowTitle('Import') 
        styleForm(Form)
        
        self.centralWidget = QWidget(Form)

        self.vlayout = QVBoxLayout()
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(2)
        self.centralWidget.setLayout(self.vlayout)


        layout = QHBoxLayout()
        self.BImport = DCDButton(self.centralWidget, 'Import')
        self.BImport.setToolTip('Import Segmentation Labels')
        self.BImport.setIcon(QIcon('icons/loadmodel.png'))
        layout.addWidget(self.BImport)

        self.CBLabelType = QLabel(self.centralWidget)
        self.CBLabelType.setText('_________________________')

        layout.addWidget(self.CBLabelType)

        self.vlayout.addLayout(layout)

        self.centralWidget.setFixedSize(self.vlayout.sizeHint())
        Form.setFixedSize(self.vlayout.sizeHint())

class ImportWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ImportWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        

        self.BImport.clicked.connect(self.ImportMasks)

    def show(self):
        self.ModeChanged()
        super(ImportWindow, self).show()

    def ModeChanged(self):

        self.BImport.setEnabled(False)
        if self.parent.LearningMode() == dlMode.Segmentation:
            self.BImport.setEnabled(True)
        self.CBLabelType.setText('  ' + self.parent.dl.Mode.type.name)
        

    def ImportMasks(self):
        folder = openFolder("Select Import Folder")
        if not folder:
            return
        files = glob.glob(os.path.join(folder,'*.tif'))
        num = len(files)
        if self.parent.ConfirmPopup('Convert %i files to segmentation masks?' %num):
            self.parent.emitinitProgress(num)
            for f in files:
                image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                cnt = extractContoursFromLabel(image)

                folder, file = os.path.split(os.path.abspath(f))
                folder_out = os.path.join(folder,self.parent.dl.LabelFolderName)
                file_out = os.path.join(folder_out,file.replace('tif', 'npz')) 


                self.parent.files.ensureFolder(folder_out)
                saveContours(cnt,file_out)
                self.parent.emitProgress()
            
            self.parent.writeStatus('%i masks successfully converted' %num)
            self.parent.emitProgressFinished()