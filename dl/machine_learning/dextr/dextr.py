#!/usr/bin/env python
import numpy as np
from scipy import misc
from tensorflow.keras import backend as K

import tensorflow as tf
import dl.machine_learning.dextr.resnet as resnet



class DEXTR(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights, num_input_channels=4,
                 classifier='psp', use_numpy=False, sigmoid=False):
        self.input_shape = input_shape
        self.num_input_channels = num_input_channels
        self.sigmoid = sigmoid
        self.model = resnet.build_network(nb_classes=nb_classes, resnet_layers=resnet_layers, num_input_channels=num_input_channels,
                                          input_shape=self.input_shape, classifier=classifier, sigmoid=self.sigmoid, output_size=self.input_shape)
        self.model.load_weights(weights)

    def predict(self, img):
        # Preprocess
        img = misc.imresize(img, self.input_shape)
        img = img.astype('float32')
        probs = self.feed_forward(img)
        return probs

    def feed_forward(self, data):
        assert data.shape == (self.input_shape[0], self.input_shape[1], self.num_input_channels)
        prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction



