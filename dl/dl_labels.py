# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:37:21 2020

@author: Koerber
"""


import numpy as np
import utils.Contour as Contour

def LoadLabel(filename, height, width):
    contours = Contour.loadContours(filename)
    label = np.zeros((height, width, 1), np.uint8)
    return Contour.drawContoursToLabel(label, contours) 