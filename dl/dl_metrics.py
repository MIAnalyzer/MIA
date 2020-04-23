
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU
import tensorflow as tf

def mean_iou_binary(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32')
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()))

def mean_iou_binary_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou_binary(y_true, y_pred)

def mean_iou(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    true_pixels = K.argmax(y_true, axis=-1) 
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    for i in range(nb_classes-1):
        true_labels = K.cast(K.equal(true_pixels, i), K.floatx())
        pred_labels = K.cast(K.equal(pred_pixels, i), K.floatx())
        intersection = K.sum(true_labels * pred_labels)
        union = K.sum(true_labels) + K.sum(pred_labels) - intersection
        res = (intersection + K.epsilon()) / (union + K.epsilon())
        iou.append(res)
    return K.mean(tf.stack(iou))


def mean_iou_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou(y_true, y_pred)


def iou_function(binary, weighted):
    #return [] # <- not running atm
    if binary:
        if weighted:
            return [mean_iou_binary_weighted]
        else:
            return [mean_iou_binary]
    else:
        if weighted:
            return [mean_iou_weighted]
        else:
            return [mean_iou]
