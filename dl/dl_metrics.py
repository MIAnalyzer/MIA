
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU
import tensorflow as tf

def mean_iou_binary(y_true, y_pred):
    # https://ai-pool.com/d/keras_iou_implementation
    
    y_pred = K.cast(K.greater(y_pred, .5), dtype='float32') # .5 is the threshold
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

def mean_iou_binary_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou_binary(y_true, y_pred)

def mean_iou(y_true, y_pred):
    # https://github.com/keras-team/keras/issues/11350

    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.dtypes.cast(K.reshape(y_true, (-1, 1))[:,0], tf.int32)
    y_true = K.one_hot(y_true, nb_classes)
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    iou = []
    flag = tf.convert_to_tensor(-1, dtype='float64')
    for i in range(nb_classes-1):
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.dtypes.cast(true_labels & pred_labels, tf.int32)
        union = tf.dtypes.cast(true_labels | pred_labels, tf.int32)
        cond = (K.sum(union) > 0) & (K.sum(tf.dtypes.cast(true_labels, tf.int32)) > 0)
        res = tf.cond(cond, lambda: K.sum(inter)/K.sum(union), lambda: flag)
        iou.append(res)
    iou = tf.stack(iou)
    legal_labels = tf.greater(iou, flag)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def mean_iou_weighted(y_true, y_pred):
    y_true = y_true[...,:-1]
    return mean_iou(y_true, y_pred)


def iou_function(binary, weighted):
    return [] # <- not running atm
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
