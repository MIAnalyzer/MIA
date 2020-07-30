
from utils.Shape import Shape, IMAGELABEL_ID
import numpy as np
from utils.Contour import packContours, unpackContours

class ImageLabel(Shape):
    def __init__(self, classlabel):
        super(ImageLabel,self).__init__(classlabel, IMAGELABEL_ID)

    def inside(self, point, maxdistance):
        return True
    

# utility
def saveImageLabel(label, filename, background = None):
    bg = packContours(background)
    if label:
        l = [label.labeltype]
        l.append(label.classlabel)
    else:
        if bg:
            l = [IMAGELABEL_ID, []]
    if bg:
        l.extend(bg)
    np.savez(filename, *l)
        
def loadImageLabel(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    ret = []
    if data[0] == IMAGELABEL_ID:
        if data[1]:
            ret = ImageLabel(data[1])
        else:
            ret = None
    if len(data) > 2:
        bg = unpackContours(data[2:])
    else:
        bg = None
    return ret, bg
