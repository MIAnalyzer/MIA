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

        
    def closeEvent(self, event):
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
                self.start()
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
            image1 = imutils.rotate(self.image1, self.rot1)

        if self.rot2 != 0:
            image2 = imutils.rotate(self.image2, self.rot2)
        
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
        
        cv2.namedWindow( "trackbar")
        cv2.createTrackbar("1 in y", "trackbar", 0, self.image1.shape[0], self.shift_1y)
        cv2.createTrackbar("1 in x", "trackbar", 0, self.image1.shape[1], self.shift_1x)
        cv2.createTrackbar("2 in y", "trackbar", 0, self.image2.shape[0], self.shift_2y)
        cv2.createTrackbar("2 in x", "trackbar", 0, self.image2.shape[1], self.shift_2x)
        cv2.setTrackbarPos("1 in y", "trackbar", self.y1_shift)
        cv2.setTrackbarPos("1 in x", "trackbar", self.x1_shift)
        cv2.setTrackbarPos("2 in y", "trackbar", self.y2_shift)
        cv2.setTrackbarPos("2 in x", "trackbar", self.x2_shift)
        
        cv2.createTrackbar("rotate1", "trackbar", 0, 180, self.rotate1)
        cv2.setTrackbarPos("rotate1", "trackbar", 90)
        cv2.createTrackbar("rotate2", "trackbar", 0, 180, self.rotate2)
        cv2.setTrackbarPos("rotate2", "trackbar", 90)
        self.shiftManual()
    
    def SemiAutoRegister(self):
        spot = self.selectCommonRoi()
        self.RegisteredImage = self.Register(spot)
        self.ShowResult()
        
    def selectCommonRoi(self):
        combined = np.zeros((max(self.image1.shape[0], self.image2.shape[0]), self.image1.shape[1] + self.image2.shape[1]), np.uint8)

        combined[0:self.image1.shape[0],0:self.image1.shape[1]] = self.image1
        combined[0:self.image2.shape[0],self.image1.shape[1]:self.image1.shape[1]+self.image2.shape[1]] = self.image2
        
        width = 1601
        height = 1201
        factor = 1 + 1/9
        while width > 1600 or height > 1200:
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
        
        while width > 1600 or height > 1200:
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
            resheight = im1_y + height1
        else:
            im1_y = 0
            im2_y = y1 - y2
            y_begin = im2_y
            y_end = height2
            resheight = im2_y + height2
        if x2 > x1:
            im1_x = x2 - x1
            im2_x = 0
            x_begin = im1_x
            x_end = width2
            reswidth = im1_x + width1
        else:
            im1_x = 0
            im2_x = x1 - x2
            x_begin = im2_x
            x_end = width1
            reswidth = im2_x + width2
            
            
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

