
import cv2
from PIL import Image
import numpy as np


def supportedImageFormats():
    return ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


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


