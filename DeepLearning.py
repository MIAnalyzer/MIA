# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:54:36 2019

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.metrics import MeanIoU

import cv2
import glob
import os
import numpy as np
from enum import Enum
import dl_data
import dl_models
import dl_losses

import Contour


class dlMode(Enum):
    classification = 1
    segmentation = 2
    # add more: tracking, 3Dsegmentation



class DeepLearning():
    def __init__(self):
        self.TrainPath = None
        self.LabelPath = None
        self.TestImagesPath = None
        self.TestLabelsPath = None
        self.TrainInMemory = True
        self.ImageScaleFactor = 0.5
        self.initialized = False
        
        self.Model = None
        self.Mode = dlMode.segmentation
        self.NumClasses = 2
        
        # Training settings
        self.MonoChrome = True
        self.batch_size = 6
        self.epochs = 2500
        self.learning_rate = 1e-4
        
        
    def initModel(self, numClasses = 2):
        self.NumClasses = numClasses
        #self.Model = dl_models.resnetSegModel(self.NumClasses, monochrome = self.MonoChrome)
        self.Model = dl_models.load_simple_model(self.NumClasses, monochrome = self.MonoChrome)
        self.initialized = True
       
    def Train(self):
        if not self.initialized or self.TrainPath is None or self.LabelPath is None:
            return
        
        if self.TrainInMemory:
            x,y = dl_data.loadTrainingDataSet(self.TrainPath, self.LabelPath, self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
            train_generator = dl_data.TrainingDataGenerator_inMemory(x,y, batch_size = self.batch_size, numclasses = self.NumClasses)
        else:
            x,y = dl_data.getTrainingDataset(self.TrainPath, self.LabelPath)
            train_generator = dl_data.TrainingDataGenerator_fromDisk(x,y, batch_size = self.batch_size, numclasses = self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
        
        adam = Adam(lr=self.learning_rate)
        if self.NumClasses > 2:
            loss = dl_losses.focal_loss
        else:
            loss = 'binary_crossentropy'
        self.Model.compile(optimizer=adam, loss=loss, metrics=[MeanIoU(num_classes=self.NumClasses)])   
        self.Model.fit_generator(train_generator, steps_per_epoch=len(x)/self.batch_size,verbose=1, callbacks=None, epochs=self.epochs, workers = 20)

       
        
    
    def Predict(self):    
        if not self.initialized or self.TestImagesPath is None:
            return
        
        images = glob.glob(os.path.join(self.TestImagesPath,'*.*'))
        images = [x for x in images if (x.endswith(".tif") or x.endswith(".bmp") or x.endswith(".jpg") or x.endswith(".png"))]
        
        for i in range (len(images)):
            print("predicted " + str(i))
            if self.MonoChrome:
                image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(images[i], cv2.IMREAD_COLOR)
                
            ### to do -> ensure matching size, either here or in the model
            width = image.shape[1]
            height = image.shape[0]
            image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))
            split_factor = 16
            pad = (split_factor - image.shape[0] % split_factor, split_factor - image.shape[1] % split_factor)
            pad = (pad[0] % split_factor, pad[1] % split_factor)            
            if pad != (0,0):
                image = np.pad(image, ((0, pad[0]), (0, pad[1])),'constant', constant_values=(0, 0))
               
            image = image[np.newaxis, :, :, np.newaxis].astype('float')/255.
            pred = self.Model.predict(image)
            name = os.path.splitext(os.path.basename(images[i]))[0]
            pred = np.squeeze(np.argmax(pred, axis = 3))
            pred = pred[0:pred.shape[0]-pad[0],0:pred.shape[1]-pad[1]]
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.TestLabelsPath, (name+ ".tif")) , pred)
            contours = Contour.extractContoursFromLabel(pred)
            Contour.saveContours(contours, os.path.join(self.TestLabelsPath, (name+ ".npz")))
        
    
    def setMode(self, mode):
        self.Mode = mode
            
    def Reset(self):
        self.initialized = False
    
    def LoadModel(self, modelpath):
        pass
    
    def SaveModel(self, modelpath):
        pass
    
    