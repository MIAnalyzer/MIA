# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:37:03 2023

@author: User
"""

# adapted from https://github.com/facebookresearch/segment-anything
# licence Apache, https://github.com/facebookresearch/segment-anything/blob/main/LICENSE

import numpy as np
from dl.machine_learning.segment_anything import sam_model_registry, SamPredictor

class SegmentAnything():
    def __init__(self):
        self.model ='vit_b'
        self.sam_vit_b_ckp = './dl/machine_learning/segment_anything/sam_vit_b_01ec64.pth'
        self.sam_vit_l_ckp = './dl/machine_learning/segment_anything/sam_vit_l_0b3195.pth'
        self.device = "cuda"
        self.initialized = False
        self.image = None
        
    def loadModel(self):
        self.sam = sam_model_registry[self.model](checkpoint=self.sam_vit_b_ckp)
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)
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
                    t =time.time()
                    self.predictor.set_image(image)
                    self.image = image
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