
import cv2
from PIL import Image
import numpy as np

def supportedImageFormats():
    return ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

class ImageFile():
    def __init__(self, path, asBGR = False):
        self.__image = None
        self.__stack = False
        self.readImage(path)
        if asBGR:
            self.normalizeImage()
            self.convertToBGR()
        
    def readImage(self, path):
        try:
            image = Image.open(path)
            frames = image.n_frames
            images = []
            for i in range(frames):
                image.seek(i)
                images.append(np.array(image))
            self.__image = np.array(images)
            # convert to bgr if color
            if len(self.__image.shape)>=4: 
                self.__image = self.__image[:, :, :, [2, 1, 0],...].copy()
                
            if frames == 1:
                self.__stack = False
                self.__image = np.squeeze(self.__image)
            elif frames > 1:
                self.__stack = True            
        except:
            self.__image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.__stack = False
        
    def convertToBGR(self):
        if self.__image is None:
            return
        if self.__stack:
            if len(self.__image.shape) == 3 or self.__image.shape[3] == 1:
                images = []
                for i in range (self.__image.shape[0]):
                    image = cv2.cvtColor(self.__image[i], cv2.COLOR_GRAY2BGR )
                    images.append(image)
                self.__image = np.array(images) 
            if len(self.__image.shape) == 5:
                images = []
                for i in range (self.__image.shape[0]):
                    image = cv2.cvtColor(self.__image[i], cv2.CV_BGRA2BGR )
                    images.append(image)
                self.__image = np.array(images) 
            return
        else:
            if len(self.__image.shape) == 2 or self.__image.shape[2] == 1:
                self.__image = cv2.cvtColor(self.__image, cv2.COLOR_GRAY2BGR )
            if len(self.__image.shape) == 4:
                self.__image = cv2.cvtColor(self.__image, cv2.CV_BGRA2BGR )
 
    def normalizeImage(self):
        if self.__image is None:
            return
        if self.__stack:
            return 
        else:
            norm_image = np.zeros(self.__image.shape)
            norm_image = cv2.normalize(self.__image,norm_image, 0, 255, cv2.NORM_MINMAX)
            self.__image = norm_image.astype('uint8')
            
    def isStack(self):
        return True if self.__stack else False
    
    def numOfImagesInStack(self):
        if not self.__stack:
            return 1
        else:
            return self.__image.shape[0]

    def getImage(self, frame = 0):
        if self.__stack:
            return self.__image[frame]
        return self.__image


def read_tiff(path):
    # returns a stack of [num, w,h,(ch)]
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def normalizeImage(image):
    norm_image = np.zeros(image.shape)
    norm_image = cv2.normalize(image,norm_image, 0, 255, cv2.NORM_MINMAX)
    image = norm_image.astype('uint8')
    return image

def convertToBGR(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.CV_BGRA2BGR )
    return image

def readImageAsBGR(path):     
    # always returns 3-channel normalized opencv image with 8-bit depth -> for displaying purposes mainly
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = normalizeImage(image)
    image = convertToBGR(image)
    return image


