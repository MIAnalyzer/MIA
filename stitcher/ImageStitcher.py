# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:03:10 2020

@author: Koerber
"""


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import imutils
import sys
import cv2
import numpy as np
import imutils
import glob
import os



class Window(object):
    def setupUi(self, Form):
        width = 400
        height= 200
        Form.setWindowTitle('Manual Shift') 
        Form.setStyleSheet("background-color: rgb(250, 250, 250)")
        
        Form.setFixedSize(width, height)
        self.centralWidget = QWidget(Form)
        self.centralWidget.setFixedSize(width, height)
        
        self.mainhlayout = QHBoxLayout(self.centralWidget)
        self.centralWidget.setLayout(self.mainhlayout)
        
        ### image1 ###
        submainvlayout = QVBoxLayout(self.centralWidget)
        submainvlayout.setSpacing(10)
        ## title
        self.I1_title = QLabel(self.centralWidget)
        self.I1_title.setText('Image 1')
        self.I1_title.setStyleSheet("font-weight: bold; color: green")
        self.I1_title.setContentsMargins(0,0,0,5)
        submainvlayout.addWidget(self.I1_title, stretch = 1,alignment= Qt.AlignHCenter)
        
        ## move horizontal ##
        h1layout = QHBoxLayout(self.centralWidget)
        self.I1_horznum = QSpinBox(self.centralWidget)
        self.I1_horznum.setFixedWidth(50)
        self.I1_horzslide = QSlider(Qt.Horizontal, self.centralWidget)
        h1layout.addWidget(self.I1_horznum,stretch=2)
        h1layout.addWidget(self.I1_horzslide,stretch=3)
        h1layout.setSpacing(5)
        h1layout.setContentsMargins(0,0,20,0)
        submainvlayout.addLayout(h1layout, stretch = 5)
        
        ## rotate + move vertical ##
        # rotate #
        h2layout = QHBoxLayout(self.centralWidget)
        v2layout = QVBoxLayout(self.centralWidget)
        self.I1_rotdial = QDial(self.centralWidget)
        h3layout = QHBoxLayout(self.centralWidget) 
        self.I1_rotnum = QSpinBox(self.centralWidget)
        self.I1_rotnum.setFixedWidth(50)
        spacer = QSpacerItem(1, 1)
        h3layout.addItem(spacer)
        h3layout.addWidget(self.I1_rotnum)
        spacer = QSpacerItem(1, 1)
        h3layout.addItem(spacer)       
        v2layout.addWidget(self.I1_rotdial)
        v2layout.addLayout(h3layout)
        h2layout.addLayout(v2layout, stretch=3)
        
        # move vertical #
        v3layout = QVBoxLayout(self.centralWidget)
        self.I1_vertslide = QSlider(Qt.Vertical, self.centralWidget)
        self.I1_vertnum = QSpinBox(self.centralWidget)
        self.I1_vertnum.setFixedWidth(50)
        v3layout.addWidget(self.I1_vertslide, alignment= Qt.AlignHCenter)
        v3layout.addWidget(self.I1_vertnum)
        v3layout.setSpacing(5)
        h2layout.addLayout(v3layout, stretch = 1)
        h2layout.setSpacing(10)
        submainvlayout.addLayout(h2layout, stretch = 10)
        
        self.mainhlayout.addItem(submainvlayout)
        line = QFrame(self.centralWidget)
        line.setFrameShape(QFrame.VLine)
        self.mainhlayout.addWidget(line)
        
        ### image2 ###
        submainvlayout = QVBoxLayout(self.centralWidget)
        submainvlayout.setSpacing(10)
        ## title
        self.I2_title = QLabel(self.centralWidget)
        self.I2_title.setText('Image 2')
        self.I2_title.setStyleSheet("font-weight: bold; color: red")
        self.I2_title.setContentsMargins(0,0,0,5)
        submainvlayout.addWidget(self.I2_title, stretch = 1,alignment= Qt.AlignHCenter)
        
        ## move horizontal ##
        h1layout = QHBoxLayout(self.centralWidget)
        self.I2_horznum = QSpinBox(self.centralWidget)
        self.I2_horznum.setFixedWidth(50)
        self.I2_horzslide = QSlider(Qt.Horizontal, self.centralWidget)
        h1layout.addWidget(self.I2_horznum,stretch=2)
        h1layout.addWidget(self.I2_horzslide,stretch=3)
        h1layout.setSpacing(5)
        h1layout.setContentsMargins(0,0,20,0)
        submainvlayout.addLayout(h1layout, stretch = 5)
        
        ## rotate + move vertical ##
        # rotate #
        h2layout = QHBoxLayout(self.centralWidget)
        v2layout = QVBoxLayout(self.centralWidget)
        self.I2_rotdial = QDial(self.centralWidget)
        h3layout = QHBoxLayout(self.centralWidget) 
        self.I2_rotnum = QSpinBox(self.centralWidget)
        self.I2_rotnum.setFixedWidth(50)
        spacer = QSpacerItem(1, 1)
        h3layout.addItem(spacer)
        h3layout.addWidget(self.I2_rotnum)
        spacer = QSpacerItem(1, 1)
        h3layout.addItem(spacer)       
        v2layout.addWidget(self.I2_rotdial)
        v2layout.addLayout(h3layout)
        h2layout.addLayout(v2layout, stretch=3)
        
        # move vertical #
        v3layout = QVBoxLayout(self.centralWidget)
        self.I2_vertslide = QSlider(Qt.Vertical, self.centralWidget)
        self.I2_vertnum = QSpinBox(self.centralWidget)
        self.I2_vertnum.setFixedWidth(50)
        v3layout.addWidget(self.I2_vertslide, alignment= Qt.AlignHCenter)
        v3layout.addWidget(self.I2_vertnum)
        v3layout.setSpacing(5)
        h2layout.addLayout(v3layout, stretch = 1)
        h2layout.setSpacing(10)
        submainvlayout.addLayout(h2layout, stretch = 10)
        
        self.mainhlayout.addItem(submainvlayout)

        

class ManualWindow(QMainWindow, Window):
    def __init__(self, parent ):
        super(ManualWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
    
    def initValues(self):
        self.I1_horznum.setRange(0,self.parent.image1.shape[1])  
        self.I1_horznum.setValue(self.parent.x1_shift)
        self.I1_vertnum.setRange(0,self.parent.image1.shape[0])
        self.I1_vertnum.setValue(self.parent.y1_shift)
        
        self.I2_horznum.setRange(0,self.parent.image2.shape[1])
        self.I2_horznum.setValue(self.parent.x2_shift)
        self.I2_vertnum.setRange(0,self.parent.image2.shape[0])
        self.I2_vertnum.setValue(self.parent.y2_shift)
        
        
        self.I1_horzslide.setMinimum(0)
        self.I1_horzslide.setMaximum(self.parent.image1.shape[1])
        self.I1_horzslide.setValue(self.parent.x1_shift)
        self.I1_vertslide.setMinimum(0)
        self.I1_vertslide.setMaximum(self.parent.image1.shape[0])
        self.I1_vertslide.setValue(self.parent.y1_shift)
        self.I1_vertslide.setInvertedAppearance(True)
        
        self.I2_horzslide.setMinimum(0)
        self.I2_horzslide.setMaximum(self.parent.image2.shape[1])
        self.I2_horzslide.setValue(self.parent.x2_shift)
        self.I2_vertslide.setMinimum(0)
        self.I2_vertslide.setMaximum(self.parent.image2.shape[0])
        self.I2_vertslide.setValue(self.parent.y2_shift)
        self.I2_vertslide.setInvertedAppearance(True)
        
        
        self.I1_rotdial.setMinimum(0)
        self.I1_rotdial.setMaximum(180)
        self.I1_rotdial.setValue(90)
        self.I1_rotnum.setRange(0,180)  
        self.I1_rotnum.setValue(90)

        self.I2_rotdial.setMinimum(0)
        self.I2_rotdial.setMaximum(180)
        self.I2_rotdial.setValue(90)
        self.I2_rotnum.setRange(0,180)  
        self.I2_rotnum.setValue(90)


        self.I1_horznum.valueChanged.connect(self.shift_1x)
        self.I1_horzslide.sliderMoved.connect(self.shift_1x_slider)
        self.I1_vertnum.valueChanged.connect(self.shift_1y)
        self.I1_vertslide.sliderMoved.connect(self.shift_1y_slider)
        self.I1_rotnum.valueChanged.connect(self.rot_1)
        self.I1_rotdial.sliderMoved.connect(self.rot_1_slider)
        
        self.I2_horznum.valueChanged.connect(self.shift_2x)
        self.I2_horzslide.sliderMoved.connect(self.shift_2x_slider)
        self.I2_vertnum.valueChanged.connect(self.shift_2y)
        self.I2_vertslide.sliderMoved.connect(self.shift_2y_slider)
        self.I2_rotnum.valueChanged.connect(self.rot_2)
        self.I2_rotdial.sliderMoved.connect(self.rot_2_slider)
    
        
    def shift_1x(self):
        self.parent.shift_1x(self.I1_horznum.value())
        self.I1_horzslide.setValue(self.I1_horznum.value())
    def shift_1x_slider(self):
        self.I1_horznum.setValue(self.I1_horzslide.value())
    def shift_1y(self):
        self.parent.shift_1y(self.I1_vertnum.value())
        self.I1_vertslide.setValue(self.I1_vertnum.value())
    def shift_1y_slider(self):    
        self.I1_vertnum.setValue(self.I1_vertslide.value())
    def rot_1(self):
        self.parent.rotate1(self.I1_rotnum.value())
        self.I1_rotdial.setValue(self.I1_rotnum.value())
    def rot_1_slider(self):
        self.I1_rotnum.setValue(self.I1_rotdial.value())
        
    def shift_2x(self):
        self.parent.shift_2x(self.I2_horznum.value())
        self.I2_horzslide.setValue(self.I2_horznum.value())
    def shift_2x_slider(self):
        self.I2_horznum.setValue(self.I2_horzslide.value())
    def shift_2y(self):
        self.parent.shift_2y(self.I2_vertnum.value())
        self.I2_vertslide.setValue(self.I2_vertnum.value())
    def shift_2y_slider(self):
        self.I2_vertnum.setValue(self.I2_vertslide.value())
    def rot_2(self):
        self.parent.rotate2(self.I2_rotnum.value())
        self.I2_rotdial.setValue(self.I2_rotnum.value())
    def rot_2_slider(self):
        self.I2_rotnum.setValue(self.I2_rotdial.value())
        
        
        
        
        

class ImageStitcher(QWidget):
    def __init__(self):
        super(ImageStitcher, self).__init__()
        
        layout = QVBoxLayout(self)
        self.RegisterAllButton = QPushButton(self)
        self.RegisterAllButton.setText('Register All')
        self.RegisterAllButton.clicked.connect(self.RegisterAll)
        
        self.RegisterButton = QPushButton(self)
        self.RegisterButton.setText('Load Images')
        self.RegisterButton.clicked.connect(self.startRegistration)
        
        self.SelectSpot = QPushButton(self)
        self.SelectSpot.setText('Select Common ROI')
        self.SelectSpot.clicked.connect(self.SemiAutoRegister)
        self.SelectSpot.setEnabled(False)
        
        layout2 = QHBoxLayout(self)
        self.ManualRButton = QPushButton(self)
        self.ManualRButton.setText('Manual')
        self.ManualRButton.clicked.connect(self.ManualRegistration)
        self.ManualRButton.setEnabled(False)
        self.FinishButton = QPushButton(self)
        self.FinishButton.setText('Show Result')
        self.FinishButton.clicked.connect(self.finishManualRegistration)
        self.FinishButton.setEnabled(False)
        layout2.addWidget(self.ManualRButton)
        layout2.addWidget(self.FinishButton)

        
        self.SaveImage = QPushButton(self)
        self.SaveImage.setText('Save Image')
        self.SaveImage.clicked.connect(self.saveImage)
        self.SaveImage.setEnabled(False)
        
        layout.addWidget(self.RegisterAllButton)
        layout.addWidget(self.RegisterButton)
        layout.addWidget(self.SelectSpot)
        layout.addLayout(layout2)
        layout.addWidget(self.SaveImage)
        
        self.manushift = ManualWindow(self)
        
        self.show()    
        self.image1 = None
        self.image2 = None
        self.RegisteredImage = None
        self.x1_shift = 0
        self.y1_shift = 0
        self.x2_shift = 0
        self.y2_shift = 0
        self.rot1 = 0
        self.rot2 = 0
        screen_resolution = QDesktopWidget().screenGeometry()
        self.maxheight = screen_resolution.height() - 200
        self.maxwidth = screen_resolution.width() - 200
        self.onlyonce = False

        
    def closeEvent(self, event): 
        self.manushift.hide()
        cv2.destroyAllWindows()

        
        
    def RegisterAll(self):
        # searches for all images in folder that end with _a.tif and _b.tif or _A.tif and _B.tif and performs their registration
        folder = str(QFileDialog.getExistingDirectory(self, "Select Image Folder"))
        if not folder:
            return
        exts = ['*.tif']
        files = [f for ext in exts for f in glob.glob(os.path.join(folder, ext))]
        files.sort()
        names = [os.path.splitext(os.path.basename(x))[0] for x in files]
        for f in names:
            if f[-1] == 'a' or f[-1] == 'A':
                match = [s for s in names if f[:-1] == s[:-1] and (s[-1] == 'b' or s[-1] == 'B')]
                if match != [] and len(match) == 1:
                    name1 = os.path.join(folder, f + '.tif')
                    name2 = os.path.join(folder, match[0] + '.tif')
                    self.image1 = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
                    self.image2 = cv2.imread(name2, cv2.IMREAD_GRAYSCALE)
                    if self.image1 is not None and self.image2 is not None:  
                        self.RegisteredImage = self.Register()
                        cv2.imwrite(os.path.join(folder, f[:-1] + 'ab.tif'), self.RegisteredImage)
        
        
    def startRegistration(self):
        filenames = QFileDialog.getOpenFileNames(None, "Select Files to Register", '',"Image (*.tif)")[0]
        if filenames == []:
            return
        if len(filenames) != 2:
            if self.OkCancelPopup('Please select 2 image files'):
                self.startRegistration()
            else:
                return            
            
        self.image1 = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE)
        self.image2 = cv2.imread(filenames[1], cv2.IMREAD_GRAYSCALE)  
        

        self.RegisteredImage = self.Register()
        self.ShowResult()

        self.ManualRButton.setEnabled(True)
        self.FinishButton.setEnabled(True)
        self.SaveImage.setEnabled(True)
        self.SelectSpot.setEnabled(True)
        
            
    def saveImage(self, image):
        filename = QFileDialog.getSaveFileName(self, "Save Registered Image", '', "Image File (*.tif)")[0]
        if filename:
            cv2.imwrite(filename, self.RegisteredImage)
    
    def shift_1y(self,x):
        self.y1_shift = x
        self.shiftManual()
    
    def shift_2y(self,x):
        self.y2_shift = x
        self.shiftManual()
        
    def shift_1x(self,x):
        self.x1_shift = x
        self.shiftManual()
    
    def shift_2x(self,x):
        self.x2_shift = x
        self.shiftManual()
        
    def rotate1(self,x):
        self.rot1 = x -90
        self.shiftManual()
        
    def rotate2(self,x):
        self.rot2 = x -90
        self.shiftManual()
        
   
    
    def finishManualRegistration(self):
        self.shiftManual(False)
    
    def shiftManual(self, colored = True):
        image1 = self.image1
        image2 = self.image2
        
        
        if self.rot1 != 0:
            image1 = imutils.rotate_bound(self.image1, self.rot1)

        if self.rot2 != 0:
            image2 = imutils.rotate_bound(self.image2, self.rot2)
        
        h1 = image1.shape[0]
        h2 = image2.shape[0]
        w1 = image1.shape[1]
        w2 = image2.shape[1]
        
        h_c = max(h1+self.y1_shift, h2 + self.y2_shift)
        w_c = max(w1+self.x1_shift, w2 + self.x2_shift)

        
        if self.y1_shift < self.y2_shift:
            y_begin = self.y2_shift
            y_end = self.y1_shift + h1
        else:
            y_begin = self.y1_shift
            y_end = self.y2_shift + h2
        if self.x1_shift < self.x2_shift:
            x_begin = self.x2_shift
            x_end = self.x1_shift + w1
        else:
            x_begin = self.x1_shift
            x_end = self.x2_shift + w2  
        
        if colored:
            manualShiftImage = np.zeros((h_c, w_c, 3), np.uint8)
            manualShiftImage[self.y1_shift:self.y1_shift+h1,self.x1_shift:self.x1_shift+w1, 1] = image1 
            manualShiftImage[self.y2_shift:self.y2_shift+h2,self.x2_shift:self.x2_shift+w2, 2] = image2
            self.SaveImage.setEnabled(False)
        else:
            manualShiftImage = np.zeros((h_c, w_c), np.uint16)
            manualShiftImage[self.y1_shift:self.y1_shift+h1,self.x1_shift:self.x1_shift+w1] = image1
            insert1 = image2
            insert2 = manualShiftImage[self.y2_shift:self.y2_shift+h2,self.x2_shift:self.x2_shift+w2]
            overlap = (insert1 + insert2)
            nonzero = np.where(np.logical_and(insert1, insert2))
            overlap[nonzero] = overlap[nonzero] /2
            
            manualShiftImage[self.y2_shift:self.y2_shift+h2,self.x2_shift:self.x2_shift+w2] = (image2 + manualShiftImage[self.y2_shift:self.y2_shift+h2,self.x2_shift:self.x2_shift+w2])
            manualShiftImage[self.y2_shift:self.y2_shift+h2,self.x2_shift:self.x2_shift+w2] = overlap
            manualShiftImage =  manualShiftImage.astype(np.uint8)
            
            
            self.SaveImage.setEnabled(True)
            
        manualShiftImage = manualShiftImage[min(self.y1_shift, self.y2_shift):,min(self.x1_shift, self.x2_shift):]
        self.RegisteredImage = manualShiftImage
        self.ShowResult()
        
          
    def ManualRegistration(self):
        self.manushift.initValues()
        self.manushift.show()
        self.shiftManual()
    
    def SemiAutoRegister(self):
        if self.onlyonce or self.OkCancelPopup('Select ROI present in both images. Press \'Enter\' to finish or \'c\' to cancel.'):
            spot = self.selectCommonRoi()
            self.onlyonce = True
        else:
            return  
        
        if spot[0] == spot[1] == spot[2] == spot[3] == 0:
            return
        self.RegisteredImage = self.Register(spot)
        self.ShowResult()
        
    def selectCommonRoi(self):
        combined = np.zeros((max(self.image1.shape[0], self.image2.shape[0]), self.image1.shape[1] + self.image2.shape[1]), np.uint8)

        combined[0:self.image1.shape[0],0:self.image1.shape[1]] = self.image1
        combined[0:self.image2.shape[0],self.image1.shape[1]:self.image1.shape[1]+self.image2.shape[1]] = self.image2
        
        width = self.maxwidth + 1
        height = self.maxheight + 1 
        factor = 1 + 1/9
        while width > self.maxwidth or height > self.maxheight:
            factor *= 0.9
            width  = combined.shape[1] * factor
            height = combined.shape[0] * factor
            
        # we have a small rounding error in factor when casting to int
        # should not matter
        image_resized = cv2.resize(combined, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        ret_us = cv2.selectROI("Select common ROI", image_resized, showCrosshair = False)
               
        ret = (int(ret_us[0] / factor), int(ret_us[1] / factor), int(ret_us[3] / factor), int(ret_us[3] / factor))
        

        lst = list(ret)
        if ret[0] > self.image1.shape[1]:
            lst[0] = ret[0] - self.image1.shape[1]
            image = self.image1
            self.image1 = self.image2
            self.image2 = image

        cv2.destroyWindow("Select common ROI")
        return lst
        
    
    def OkCancelPopup(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Continue")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        ret = msg.exec_()
        if ret == 1024:
            return True
            self.start()
        else:
            return False
        
    def ShowResult(self):
        image = self.RegisteredImage
        width = image.shape[1]
        height = image.shape[0]
        
        while width > self.maxwidth or height > self.maxheight:
            width = width * 0.9
            height = height *0.9
        
        image1 = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Result", image1)

        
        
    def Register(self, spot=None):
        
        width1 = self.image1.shape[1]
        height1 = self.image1.shape[0]
        
        width2 = self.image2.shape[1]
        height2 = self.image2.shape[0]
        
        if spot is None:
            x1 = width1//2-width1//8
            y1 = height1//2-height1//8
            h = height1 // 4
            w = width1 // 4
        else:
            x1 = spot[0]
            y1 = spot[1]
            h = spot[2]
            w = spot[3]

        crop = self.image1[y1:y1+h, x1:x1+w]
            
        match = cv2.matchTemplate(self.image2, crop, cv2.TM_CCORR_NORMED)
        _, _, _, maxLoc = cv2.minMaxLoc(match)
        
        x2 = maxLoc[0]
        y2 = maxLoc[1]

            
        if y2 > y1:
            im1_y = y2 - y1
            im2_y = 0
            y_begin = im1_y
            y_end = height2
        else:
            im1_y = 0
            im2_y = y1 - y2
            y_begin = im2_y
            y_end = height2

        if x2 > x1:
            im1_x = x2 - x1
            im2_x = 0
            x_begin = im1_x
            x_end = width2
        else:
            im1_x = 0
            im2_x = x1 - x2
            x_begin = im2_x
            x_end = width1

            
        reswidth = max(im2_x + width2,im1_x + width1)  
        resheight = max(im2_y + height2,im1_y + height1) 
        result = np.zeros((resheight, reswidth), np.uint16)

        result[im1_y:im1_y+height1,im1_x:im1_x+width1] = self.image1
        result[im2_y:im2_y+height2,im2_x:im2_x+width2] = (self.image2 + result[im2_y:im2_y+height2,im2_x:im2_x+width2])
        result[y_begin:y_end,x_begin:x_end] = result[y_begin:y_end,x_begin:x_end] / 2
        result = result.astype(np.uint8)        

        self.y1_shift = y2
        self.y2_shift = y1
        self.x1_shift = x2
        self.x2_shift = x1

        return result
        

if __name__ == "__main__":
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    stitcher = ImageStitcher()
    app.exec_()

