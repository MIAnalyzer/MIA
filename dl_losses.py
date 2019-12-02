# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 08:58:49 2019

@author: Koerber
"""




import tensorflow.keras.backend as K



#from sklearn.utils import class_weight
#        self.class_weights = {0: 1.,           
#                        1: 15.,                  
#                        2: 15.}

# implement custom loss with weights
#self.class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)



def focal_loss(ytrue, ypred, gamma=2):
    ypred /= K.sum(ypred, axis=-1, keepdims=True)
    eps = K.epsilon()
    ypred = K.clip(ypred, eps, 1. - eps)
    return -K.sum(K.pow(1. - ypred, gamma) * ytrue * K.log(ypred), axis=-1)


def ignore_unknown_xentropy(ytrue, ypred):
    return (1-ytrue[:, :, :, 0])*K.categorical_crossentropy(ytrue, ypred)