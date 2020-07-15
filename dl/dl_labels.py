# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:37:21 2020

@author: Koerber
"""


import numpy as np
import utils.Contour as Contour

import glob
import os
from itertools import repeat, chain

def LoadLabel(filename, height, width):
    contours = Contour.loadContours(filename)
    label = np.zeros((height, width, 1), np.uint8)
    return Contour.drawContoursToLabel(label, contours) 


def getAllImageLabelPairPaths(imagepath, labelpath):
    images = glob.glob(os.path.join(imagepath,'*.*'))
    labels = glob.glob(os.path.join(labelpath,'*.*'))
          
    # to do: check if images in one of supported image formats
    labels = [x for x in labels if x.endswith(".npy") or x.endswith(".npz")]
    
    image_names = [os.path.splitext(os.path.basename(each))[0] for each in images]
        
    # adds subfolders to labels as image stack labels are saved in folders
    labels.extend([os.path.join(labelpath, f) for f in os.listdir(labelpath) if (os.path.isdir(os.path.join(labelpath, f)) and f in ''.join(os.listdir(os.path.join(labelpath, f))))])
    label_names = [os.path.splitext(os.path.basename(each))[0] for each in labels]

    intersection = list(set(image_names).intersection(label_names))
    if intersection == list():
        return None, None
        

    images = [each for each in images if os.path.splitext(os.path.basename(each))[0] in intersection]
    labels = [each for each in labels if os.path.splitext(os.path.basename(each))[0] in intersection]
    images.sort()
    labels.sort()
    
    return images, labels

def splitStackLabels(image, folder):
    if not os.path.isdir(os.path.join(folder)):
        return [image], [folder]
    subfiles = [(os.path.join(folder,f),f[-7:-4]) for f in os.listdir(folder) if os.path.splitext(os.path.basename(folder))[0] in f]
    images = list(repeat((image,'stack'), len(subfiles)))
    return images, subfiles

def unrollPaths(images, labels):
    split = [splitStackLabels(x,y) for x,y in zip(images, labels)]
    images,labels = zip(*split)
    images = list(chain.from_iterable(images))
    labels = list(chain.from_iterable(labels))
    return images, labels