
import cv2
from PIL import Image
import numpy as np

def supportedImageFormats():
    return ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

class ImageFile():
    def __init__(self, path, asBGR = False):
        self._image = None
        self._stack = False
        self.readImage(path)
        self.path = path
        self.brightness = 0
        self.contrast = 1
        if asBGR:
            self.normalizeImage(force8bit = True)
            self.convertToBGR()

    def readImage(self, path):
        try: # open stack
            image = Image.open(path)
            image.seek(0)

            frames = image.n_frames
            images = []
            for i in range(frames):
                image.seek(i)
                images.append(np.array(image))

            self._image = np.array(images)
            # convert to bgr if color
            if len(self._image.shape)>=4: 
                self._image = self._image[:, :, :, [2, 1, 0],...].copy()
                
            if frames == 1:
                self._stack = False
                self._image = np.squeeze(self._image)
            elif frames > 1:
                self._stack = True            
        except:
            self._image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self._stack = False
        
    def convertToBGR(self):
        if self._image is None:
            return
        if self._stack:
            if len(self._image.shape) == 3 or self._image.shape[3] == 1:
                images = []
                for i in range (self._image.shape[0]):
                    image = cv2.cvtColor(self._image[i], cv2.COLOR_GRAY2BGR )
                    images.append(image)
                self._image = np.array(images) 
            if len(self._image.shape) == 5:
                images = []
                for i in range (self._image.shape[0]):
                    image = cv2.cvtColor(self._image[i], cv2.CV_BGRA2BGR )
                    images.append(image)
                self._image = np.array(images) 
            return
        else:
            if len(self._image.shape) == 2 or self._image.shape[2] == 1:
                self._image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR )
            if self._image.shape[2] == 4:
                self._image = cv2.cvtColor(self._image, cv2.COLOR_BGRA2BGR )
 
    def normalizeImage(self, force8bit = False):
        if self._image is None:
            return

        min_ = np.min(self._image)
        max_ = np.max(self._image)
        if self._image.dtype == np.uint16 and not force8bit:
            self._image = (self._image - min_)/(max_-min_) * 65536
            self._image = self._image.astype('uint16')
        else:
            self._image = (self._image - min_)/(max_-min_) * 255
            self._image = self._image.astype('uint8')
            
    def width(self):
        ch = 2 if self._stack else 1
        return self._image.shape[ch]
        
    def height(self):
        ch = 1 if self._stack else 0
        return self._image.shape[ch]
            
    def isStack(self):
        return True if self._stack else False
    
    def numOfImagesInStack(self):
        if not self._stack:
            return 1
        else:
            return self._image.shape[0]

    def convert2DeepLearningInput(self, monochrome, image):
        if len(image.shape) == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR )
        if len(image.shape) == 2:
            if not monochrome:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
            else:
                image = image[..., np.newaxis]
        elif len(image.shape) == 3:
            if monochrome and image.shape[2] > 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
                # atm I dont know if bgr2gray squeezes
                if len(image.shape) == 2:
                    image = image[..., np.newaxis]    
        return image
   
    def adjustBrightnessContrast(self, image):
        if self.contrast == 1 and self.brightness == 0:
            return image
        # in general this can be done more efficient and without the memory use of a 16 bit image
        im = (image.astype(np.int16) * self.contrast) + self.brightness
        im[im<0] = 0
        im[im>255]=255
        return im.astype(np.uint8)

    def getDLInputImage(self,monochrome,frame=0):
        return self.convert2DeepLearningInput(monochrome, self.getImage(frame))

    def getCorrectedImage(self, frame = 0):
        return self.adjustBrightnessContrast(self.getImage(frame))

    def getImage(self, frame = 0):
        if self._stack:
            return self._image[frame]
        else:
            return self._image


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

def readNumOfImageFrames(path):
    try:
        image = Image.open(path)
        image.seek(0)
        return image.n_frames
    except:
        return 1