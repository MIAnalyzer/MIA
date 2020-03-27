
import numpy as np
from skimage.measure import label
import numpy as np
import cv2


def createWeightedBorderMapFromLabel(y, w0 = 10, sigma = 5):

    """
    adopted from https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras

    """
    # to do make this possible for batches of size (n,h,w,1)


    weighting = {
        0: 1, # background
        1: 1  # objects
    }
    # we only differntiate between some object and background here
    y = np.squeeze(y)
    y[y >= 1] = 1

    labels = label(y)
    masks = (y > 0).astype(int)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            # opencv is faster than skimage
            c = np.zeros(y.shape)
            c [labels != label_id] = 1
            distances[:,:,i] = cv2.distanceTransform(c.astype(np.uint8),cv2.DIST_L2, cv2.DIST_MASK_PRECISE )

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if weighting:
        class_weights = np.zeros_like(y)
        for k, v in weighting.items():
            class_weights[y == k] = v
        w = w + class_weights

    return w[...,np.newaxis].astype(np.uint8)