# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:54:36 2019

@author: Koerber
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model

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
        self.TrainInMemory = True
        self.ImageScaleFactor = 0.25
        self.initialized = False
        
        self.Model = None
        self.Mode = dlMode.segmentation
        self.NumClasses = 2
        
        # Training settings
        self.MonoChrome = True
        self.batch_size = 4
        self.epochs = 100
        self.learning_rate = 1e-5
        
        
    def initModel(self, numClasses = 2):
        self.NumClasses = numClasses
        #self.Model = dl_models.resnetSegModel(self.NumClasses, monochrome = self.MonoChrome)
        self.Model = dl_models.load_simple_model(self.NumClasses, monochrome = self.MonoChrome)
        self.initialized = True
       
    def Train(self, trainingimages_path, traininglabels_path):
        if not self.initialized or trainingimages_path is None or traininglabels_path is None:
            return
        
        if self.TrainInMemory:
            x,y = dl_data.loadTrainingDataSet(trainingimages_path, traininglabels_path, self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
            train_generator = dl_data.TrainingDataGenerator_inMemory(x,y, batch_size = self.batch_size, numclasses = self.NumClasses)
        else:
            x,y = dl_data.getTrainingDataset(trainingimages_path, traininglabels_path)
            train_generator = dl_data.TrainingDataGenerator_fromDisk(x,y, batch_size = self.batch_size, numclasses = self.NumClasses, scalefactor=self.ImageScaleFactor, monochrome = self.MonoChrome)
        
        adam = Adam(lr=self.learning_rate)
        if self.NumClasses > 2:
            loss = dl_losses.focal_loss
        else:
            loss = 'binary_crossentropy'
        self.Model.compile(optimizer=adam, loss=loss, metrics=[MeanIoU(num_classes=self.NumClasses)])   
        
        ## this needs more investigation for some reasons, fit_generator is much slower than fit
        self.Model.fit_generator(train_generator, steps_per_epoch=len(x)/self.batch_size,verbose=1, callbacks=None, epochs=self.epochs, workers = 20)

       
        
    
    def PredictImages(self, image_path, label_path):    
        if not self.initialized or imagepath is None or labelpath is None:
            return
        
        images = glob.glob(os.path.join(imagepath,'*.*'))
        images = [x for x in images if (x.endswith(".tif") or x.endswith(".bmp") or x.endswith(".jpg") or x.endswith(".png"))]
        
        for i in range (len(images)):
            print("predicted " + str(i))
            if self.MonoChrome:
                image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(images[i], cv2.IMREAD_COLOR)
                
            pred = self.Predict(image)
            
            cv2.imwrite(os.path.join(labelpath, (name+ ".tif")) , pred)
            contours = Contour.extractContoursFromLabel(pred)
            Contour.saveContours(contours, os.path.join(labelpath, (name+ ".npz")))
            
    def PredictImage(self, image):    
        if not self.initialized or image is None:
            return

        ### to do -> ensure matching size, either here or in the model
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (int(width*self.ImageScaleFactor), int(height*self.ImageScaleFactor)))
        split_factor = 16
        pad = (split_factor - image.shape[0] % split_factor, split_factor - image.shape[1] % split_factor)
        pad = (pad[0] % split_factor, pad[1] % split_factor)            
        if pad != (0,0):
            image = np.pad(image, ((0, pad[0]), (0, pad[1])),'constant', constant_values=(0, 0))
                       
                        
        # to do -> handle colored images
        if self.MonoChrome:
            image = image[np.newaxis, :, :, np.newaxis].astype('float')/255.
        else:
            image = image[np.newaxis, :, :, :].astype('float')/255.
            pred = self.Model.predict(image)
            
        pred = self.Model.predict(image)
        if self.NumClasses > 2:
            pred = np.squeeze(np.argmax(pred, axis = 3))
        else:
            pred = np.squeeze(pred)
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
        pred = pred[0:pred.shape[0]-pad[0],0:pred.shape[1]-pad[1]]
        pred = pred.astype('uint8')
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        return pred

        
    
    def setMode(self, mode):
        self.Mode = mode
            
    def Reset(self):
        self.initialized = False
    
    def LoadModel(self, modelpath):
        self.Model = load_model (modelpath)
        if self.Model:
            self.initialized = True
    
    def SaveModel(self, modelpath):
        if self.initialized:
            self.Model.save(modelpath)
    
    