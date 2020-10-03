# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:36:28 2020

@author: Koerber
"""


import cv2
import numpy as np

class GrabCutSegmentation():
    def __init__(self):
        self.bg_Model = np.zeros((1,65),np.float64)
        self.fg_Model = np.zeros((1,65),np.float64)
        self.iterations = 1

    
    def trainAndSegment(self,image, mask):
        if np.any(mask == 1):
            # no samples       
            mask, self.bg_Model, self.fg_Model = cv2.grabCut(image,mask,None,self.bg_Model,self.fg_Model,self.iterations,cv2.GC_INIT_WITH_MASK)
        retmask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        return retmask

    def Segment(self,image):
        mask = np.zeros(image.shape[:2],np.uint8)
        mask[mask==0] = 2
        mask, self.bg_Model, self.fg_Model = cv2.grabCut(image,mask,None,self.bg_Model,self.fg_Model,1,cv2.GC_EVAL_FREEZE_MODEL)
        retmask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        return retmask