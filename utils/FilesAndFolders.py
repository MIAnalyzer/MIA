# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:30:13 2020

@author: Koerber
"""

from utils.Image import supportedImageFormats
import glob
import os

class FilesAndFolders():
    def __init__(self, parent):
        self.parent = parent
        
        self.trainImagespath = None
        self.testImagespath = None
        
        self.train_test_dir = True # True = TrainFolder; False = TestFolder
        
        self.files = None
        self.currentImage = -1
        self.numofImages = 0
        self.currentFrame = 0
        
    @property
    def imagepath(self): # naming unclear
        return self.trainImagespath if self.train_test_dir else self.testImagespath
        
    @property
    def labelpath(self): # naming unclear
        path = self.trainImagespath if self.train_test_dir else self.testImagespath
        if not path:
            return
        path = path + os.path.sep + self.parent.LearningMode().name + "_labels"
        if self.parent.hasStack():
            path = self.extendLabelPathByFolder(path, self.CurrentImageName()) 
        return path
    
    @property
    def trainImageLabelspath(self):
        if self.trainImagespath:
            return self.trainImagespath + os.path.sep + self.parent.LearningMode().name + "_labels"
       
    @property
    def testImageLabelspath(self):
        if self.testImagespath:
            return self.testImagespath + os.path.sep + self.parent.LearningMode().name + "_labels"
        
    # @property
    def CurrentImageName(self): 
        if self.CurrentImagePath() is None:
            return None
        return self.getFilenameFromPath(self.CurrentImagePath())

    # @property
    def CurrentLabelName(self):
        if self.parent.hasStack():
            return self.extendNameByFrameNumber(self.CurrentImageName(), self.currentFrame)
        else:
            return self.CurrentImageName()
    # @property
    def CurrentImagePath(self):
        if self.files is None:
            return None
        return self.files[self.currentImage]
        
    # @property
    def CurrentLabelPath(self):
        if self.labelpath is None or self.CurrentImageName() is None:
            return None
        path = os.path.join(self.labelpath, self.CurrentImageName())
        if self.parent.hasStack() and self.parent.separateStackLabels:
            return self.extendNameByFrameNumber(path, self.currentFrame) + ".npz"
        else:
            return path + ".npz"

    @property
    def TestImages(self):
        images = glob.glob(os.path.join(self.testImagespath,'*.*'))
        images = [x for x in images if (x.lower().endswith(tuple(supportedImageFormats())))]
        return images

    def nextImage(self):
        self.currentImage += 1
        if (self.currentImage >= self.numofImages):
            self.currentImage = 0
    
    def previousImage(self):
        self.currentImage -= 1
        if (self.currentImage < 0):
            self.currentImage = self.numofImages - 1
            
    def setTrainFolder(self, folder):
        self.trainImagespath = folder
        self.train_test_dir = True
        
    def setTestFolder(self, folder):
        self.testImagespath = folder   
        self.train_test_dir = False
     
    def switchToTrainFolder(self):
        self.train_test_dir = True
        
    def switchToTestFolder(self):
        self.train_test_dir = False          
        
    def ensureLabelFolder(self):
        self.ensureFolder(self.labelpath)
        
    def getFiles(self):
        if self.imagepath is None:
            return
        files = glob.glob(os.path.join(self.imagepath, '*.*'))          
        self.files = [x for x in files if x.lower().endswith(tuple(supportedImageFormats()))]
        if not self.files:
            self.currentImage = -1
            self.files = None
            self.numofImages = 0
        else:
            self.files.sort()
            self.currentImage = 0
            self.numofImages = len(self.files)
        
    ## helper functions
    def getFilenameFromPath(self, path):
        return os.path.splitext(os.path.basename(path))[0]
        
    def extendLabelPathByFolder(self, path, folder):
        return os.path.join(path, folder) 

    def extendNameByFrameNumber(self,name, frame):
        return name + '_' + "{0:0=3d}".format(frame)
         
    def ensureFolder(self, foldername):
        if not os.path.isdir(foldername):
            os.makedirs(foldername)
            # os.mkdir(foldername)