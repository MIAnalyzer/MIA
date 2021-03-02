
from utils.shapes.Shape import Shape, IMAGELABEL_ID
import numpy as np
from utils.shapes.Contour import packContours, unpackContours, drawbackgroundToLabel, BACKGROUNDCLASS

class ImageLabel(Shape):
    def __init__(self, classlabel):
        super(ImageLabel,self).__init__(classlabel, IMAGELABEL_ID)

    def inside(self, point, maxdistance):
        return True

    def getPosition(self):
        return None
    

# utility
def saveImageLabel(label, filename, background = None):
    bg = packContours(background)
    if label:
        l = [label.labeltype]
        l.append(label.classlabel)
    else:
        # not saving background only, anymore
        return
    #else:
    #    if bg:
    #        l = [IMAGELABEL_ID, []]
    if bg:
        l.extend(bg)
    np.savez(filename, *l)
        
def loadImageLabel(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    ret = []
    if data[0] == IMAGELABEL_ID:
        if data[1] or data[1] == 0:
            ret = ImageLabel(data[1])
        else:
            ret = None
    if len(data) > 2:
        bg = unpackContours(data[2:])
    else:
        bg = None
    return ret, bg

def drawImageLabel(image, imagelabel, bg):
    image = drawbackgroundToLabel(image, bg)
    label = BACKGROUNDCLASS if imagelabel.classlabel == 0 else imagelabel.classlabel
    if np.all(image) != 0:
        image[:] = label
    else:
        image[image==0] = label
    return image

def extractImageLabel(prediction):
    return ImageLabel(prediction)