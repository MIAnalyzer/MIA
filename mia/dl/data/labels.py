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

from enum import Enum



class dlStackLabels(Enum):
    separated = 1
    unique_collectiveInput = 2
    unique_detachedInput = 3

def getMatchingImageLabelPairsRecursive(imagepath, labelfoldername, stacklabels=dlStackLabels.separated):
    images = []
    labels = []
    for root, dirs, files in os.walk(imagepath): 
        if labelfoldername in dirs:
            # this is inefficent because we search again content of subdir, which is the information we already have in files or dirs+files
            img, lab = getMatchingImageLabelPairPaths(root,os.path.join(root,labelfoldername),stacklabels)
            if img and lab:
                images.extend(img)
                labels.extend(lab)
                
    return images, labels

def getMatchingImageLabelPairPaths(imagepath, labelpath, stacklabels=dlStackLabels.separated, unroll=True):
    if not imagepath or not labelpath:
        return list(), list()
    if not os.path.isdir(imagepath) or not os.path.isdir(labelpath):
        return list(), list()


    images = glob.glob(os.path.join(imagepath,'*.*'))
    labels = glob.glob(os.path.join(labelpath,'*.*'))
              
    labels = [x for x in labels if x.endswith(".npz")]
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
        return unrollPaths(images, labels, stacklabels)
    else:
        return images, labels

def splitStackLabels(image, folder, stacklabels):
    if not os.path.isdir(os.path.join(folder)):
        return [image], [folder]

    if stacklabels == dlStackLabels.unique_collectiveInput:
        labelname = getUniqueStackLabelFromFolderPath(folder)

        if labelname:
            label = (labelname, '-1')
            image = (image, dlStackLabels.unique_collectiveInput)
            return [image], [label]
        else:
            return [], []

    elif stacklabels == dlStackLabels.unique_detachedInput:
        labelname = getUniqueStackLabelFromFolderPath(folder)
        if labelname:
            maxframes = getFrameFromLabelPath(labelname, labeltype = stacklabels)
            images = list(repeat((image, dlStackLabels.separated), maxframes))
            labels = [(labelname,i) for i in range(maxframes)]
            return images, labels
        else:
            return [], []
        

    elif stacklabels == dlStackLabels.separated:
        subfiles = [(os.path.join(folder,f),f[-7:-4]) for f in os.listdir(folder) if os.path.splitext(os.path.basename(folder))[0]+f[-8:-4] in f]
        images = list(repeat((image, dlStackLabels.separated), len(subfiles)))
        return images, subfiles
    else:
        return [], []

def unrollPaths(images, labels, stacklabel):
    if not images or not labels:
        return None, None
    split = [splitStackLabels(x,y,stacklabel) for x,y in zip(images, labels)]
    images,labels = zip(*split)
    images = list(chain.from_iterable(images))
    labels = list(chain.from_iterable(labels))
    return images, labels

def getUniqueStackLabelFromFolderPath(folder):
    files = os.listdir(folder)
    labels = [f for f in files if (f.startswith(os.path.splitext(os.path.basename(folder))[0]) and f.endswith('u.npz'))]
    if len(labels) == 1:
        return os.path.join(folder, labels[0])
    else:
        return None
        

def CheckIfStackLabel(path):
    if isinstance(path,tuple):
        return True
    else:
        return False

def extendLabelNameByFrame(labelname, frame):
    return labelname + '_' + "{0:0=3d}".format(frame)

def extendLabelNameByMaxFrame(labelname, maxframe):
    return labelname + '_' + "{0:0=3d}".format(maxframe) + 'u'

def removeFrameFromLabelName(labelname, labeltype = dlStackLabels.separated):
    if  labeltype == dlStackLabels.separated:
        return labelname[:-4]
    else:
        return labelname[:-5]

def getFrameFromLabelPath(labelpath, labeltype = dlStackLabels.separated):
    if  labeltype == dlStackLabels.separated:
        return int(labelpath[-7:-4])
    else:
        return int(labelpath[-8:-5])