
from utils.Shape import Shape, IMAGELABEL_ID
import numpy as np

class ImageLabel(Shape):
    def __init__(self, classlabel):
        super(ImageLabel,self).__init__(classlabel, IMAGELABEL_ID)

    def inside(self, point, maxdistance):
        return True
    

# utility
def saveImageLabel(label, filename):
    l = [label.labeltype]
    l.append(label.classlabel)
    np.savez(filename, *l)
        
def loadImageLabel(filename):
    container = np.load(filename)
    data = [container[key] for key in container]
    
    ret = []
    if data[0] == IMAGELABEL_ID:
        ret = ImageLabel(data[1])
    return ret