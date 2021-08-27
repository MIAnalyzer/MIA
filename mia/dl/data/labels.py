# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:37:21 2020

@author: Koerber
"""


# import utils.Contour as Contour
# import utils.Point as Point

import glob
import os
from itertools import repeat, chain
from utils.Image import supportedImageFormats

def getMatchingImageLabelPairsRecursive(imagepath, labelfoldername):
    images = []
    labels = []
    for root, dirs, files in os.walk(imagepath): 
        if labelfoldername in dirs:
            # this is inefficent because we search again content of subdir, which is the information we already have in files or dirs+files
            img, lab = getMatchingImageLabelPairPaths(root,os.path.join(root,labelfoldername))
            if img and lab:
                images.extend(img)
                labels.extend(lab)
                
    return images, labels

def getMatchingImageLabelPairPaths(imagepath, labelpath, unroll=True):
    if not imagepath or not labelpath:
        return list(), list()
    if not os.path.isdir(imagepath) or not os.path.isdir(labelpath):
        return list(), list()

    images = glob.glob(os.path.join(imagepath,'*.*'))
    labels = glob.glob(os.path.join(labelpath,'*.*'))
              
    labels = [x for x in labels if x.lower().endswith(".npy") or x.endswith(".npz")]
    images = [x for x in images if x.lower().endswith(tuple(supportedImageFormats()))]
    
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
    
    if unroll:
        return unrollPaths(images, labels)
    else:
        return images, labels

def splitStackLabels(image, folder):
    if not os.path.isdir(os.path.join(folder)):
        return [image], [folder]
    # using same label for complete stack unhandled atm
    subfiles = [(os.path.join(folder,f),f[-7:-4]) for f in os.listdir(folder) if os.path.splitext(os.path.basename(folder))[0] in f]
    images = list(repeat((image,'stack'), len(subfiles)))
    return images, subfiles

def unrollPaths(images, labels):
    if not images or not labels:
        return None, None
    split = [splitStackLabels(x,y) for x,y in zip(images, labels)]
    images,labels = zip(*split)
    images = list(chain.from_iterable(images))
    labels = list(chain.from_iterable(labels))

    return images, labels