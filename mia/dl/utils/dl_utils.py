
import numpy as np
from skimage.measure import label
import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import utils.shapes.Contour as Contour


def createWeightedBorderMapFromLabel(batch, weighting = None, w0 = 10, sigma = 5):
    #adopted from https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
    #input: batch of size (bs,h,w,1)
    #e.g. weighting = {
    #    0: 1,   # background
    #    1: 4,   # class 1
    #    2: 1    # class 2
    #}



    out = np.zeros(batch[...,0].shape)
    for i_num in range(batch.shape[0]):
        y = batch[i_num,...].squeeze()
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
            out[i_num,...] = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        else:
            out[i_num,...] = np.zeros_like(y)
        if weighting:
            class_weights = np.zeros_like(y)
            for k, v in weighting.items():
                class_weights[y == k] = v
        else:
            class_weights = np.ones_like(y)
        out[i_num,...] = out[i_num,...] + class_weights

    return out[...,np.newaxis].astype(np.uint8)


def separatePredictions(prediction, min_distance = 20, threshold = 0.5, splitclasses = []):  
    # binary
    if len(prediction.shape) == 2 or prediction.shape[2] == 1:
        prediction_prob = prediction.copy()
        prediction[prediction>threshold] = 1
        prediction[prediction<=threshold] = 0  
        
        _, thresh = cv2.threshold(prediction.astype(np.uint8),0,255,cv2.THRESH_BINARY)
        dist_transform = distance_transform_edt(thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        labelmask = np.zeros_like(thresh, dtype=np.uint16) 
        counter = 1
        for c in contours:
            cv2.drawContours(labelmask, [c], -1, counter,-1)
            counter += 1
        
        # for some reason peaks with min_distance from image border are not detected
        # potential solution would be to extend the image
        coords = peak_local_max(dist_transform, min_distance=min_distance, labels=labelmask)
        mask = np.zeros_like(thresh, dtype=bool)
        mask[tuple(coords.T)] = True
    
        markers = ndimage.label(mask)[0]
        
        labels = watershed(-prediction_prob, markers, watershed_line=True, mask=prediction)
        
        cnt = Contour.extractContoursFromLabel(labels)
        label = np.zeros(thresh.shape, dtype=np.uint8)
        Contour.drawContoursToImage(label, cnt, separate=True)
        label[thresh==0] = 0
        label[label>1] = 1
    # multiclass 
    else:
        prediction_prob = prediction.copy()
        prediction = np.squeeze(np.argmax(prediction, axis = 2))
        label = np.zeros_like(prediction, dtype=np.uint8)
        if splitclasses == []:
            splitclasses = range(1,np.max(prediction)+1)
        else:
            for i in range(1,np.max(prediction)+1):
                if i not in splitclasses:
                    label[prediction == i] = i
        for i in splitclasses:
            pred = prediction == i
            _, thresh = cv2.threshold(pred.astype(np.uint8),0,255,cv2.THRESH_BINARY)
            dist_transform = distance_transform_edt(thresh)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            labelmask = np.zeros_like(thresh, dtype=np.uint16) 
            counter = 1
            for c in contours:
                cv2.drawContours(labelmask, [c], -1, counter,-1)
                counter += 1

            # for some reason peaks with min_distance from image border are not detected
            # potential solution would be to extend the image
            coords = peak_local_max(dist_transform, min_distance=min_distance, labels=labelmask)
            mask = np.zeros_like(thresh, dtype=bool)
            mask[tuple(coords.T)] = True
        
            markers = ndimage.label(mask)[0]
            
            labels = watershed(-prediction_prob[...,i], markers, watershed_line=True, mask=pred)
            
            cnt = Contour.extractContoursFromLabel(labels)
            label_i = np.zeros(thresh.shape, dtype=np.uint8)
            Contour.drawContoursToImage(label_i, cnt, separate=True)
            label_i[thresh==0] = 0
            label[label_i>=1] = i

    return label
    

def erodeDilate(mask, repeats, splitclasses = []):  
    ############ warning, this function was never tested ####################
    mask = mask.astype(np.uint8)
    label = np.zeros_like(mask, dtype=np.uint8)
    if splitclasses == []:
        splitclasses = range(1,np.max(mask)+1)
    else:
        for i in range(1,np.max(mask)+1):
            if i not in splitclasses:
                label[mask == i] = i
    for i in splitclasses:            
              
        pred = np.zeros_like(mask, dtype=np.uint8)
        pred[mask == i] = 1
        M = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        pred = cv2.erode(pred, M, iterations = repeats)

        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        labelmask = np.zeros_like(mask, dtype=np.uint16) 
        counter = 1
        for c in contours:
            cv2.drawContours(labelmask, [c], -1, counter,-1)
            counter += 1

        res = cv2.dilate(labelmask, M, iterations = repeats +1)
        
        total_contours = []
        for i in range(0,np.max(labelmask), 256):
            tofind = np.zeros_like(res, dtype=np.uint8)
            tofind[res>i and res < i + 256] = res - i
            contours, _ = cv2.findContours(tofind, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
            total_contours.append(contours)
        
        label_i = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(label_i, total_contours, -1, i,-1)
        label_i[mask==0] = 0


    return label