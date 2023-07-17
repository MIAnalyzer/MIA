# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:03 2023

@author: User
"""

# adapted from https://github.com/facebookresearch/segment-anything
# licence Apache, https://github.com/facebookresearch/segment-anything/blob/main/LICENSE

import cv2
import numpy as np
from dl.machine_learning.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from dl.utils.dl_downloads import get_sam

class SegmentAnything():
    def __init__(self):
        self.model ='vit_b'
        self.device = "cuda"
        self.initialized = False
        self.image = None

    def setModel(self,modelname):
        if modelname != self.model:
            self.model = modelname
            self.initialized = False
            self.image = None

        
    def loadModel(self):
        ckpt = get_sam(self.model)
        self.sam = sam_model_registry[self.model](checkpoint=ckpt)
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.initialized = True
        
        
    def setImage(self, image):
        try:
            if not self.initialized:
                self.loadModel()
                
            if self.image is None:
                self.predictor.set_image(image)
                self.image = image
            else:
                if self.image.shape != image.shape or not (self.image==image).all():
                    self.predictor.set_image(image)
                    self.image = image
        except:
            raise

    def SegmentImage(self, image):
        try:
            if not self.initialized:
                self.loadModel()
            mask_image = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
            masks = self.mask_generator.generate(image)
            contours = []
            for mask in masks:
                m = mask['segmentation']
                contour, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
                contours.append(contour[0])
            
            for c in contours:
                cv2.drawContours(mask_image, [c], -1, (0), 3)  
                cv2.drawContours(mask_image, [c], -1, (1), -1) 

            return mask_image
        except:
            raise
        
    def getMask(self, bbox, points, labels):
        mask, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            multimask_output=False,
            )
        return mask