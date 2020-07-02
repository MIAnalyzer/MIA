# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:23:05 2020

@author: Koerber
"""
import glob
import os

def getAllImageLabelPairPaths(imagepath, labelpath):
    images = glob.glob(os.path.join(imagepath,'*.*'))
    labels = glob.glob(os.path.join(labelpath,'*.*'))
          
    # to do: check if images in one of supported image formats
    labels = [x for x in labels if x.endswith(".npy") or x.endswith(".npz")]
        
    image_names = [os.path.splitext(os.path.basename(each))[0] for each in images]
    label_names = [os.path.splitext(os.path.basename(each))[0] for each in labels]
        
    intersection = list(set(image_names).intersection(label_names))
    if intersection == list():
        return None, None
        
    images = [each for each in images if os.path.splitext(os.path.basename(each))[0] in intersection]
    labels = [each for each in labels if os.path.splitext(os.path.basename(each))[0] in intersection]
    images.sort()
    labels.sort()

    return images, labels