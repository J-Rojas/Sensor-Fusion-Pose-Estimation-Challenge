
# coding: utf-8

import os
import numpy as np
from numpy import random

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Input, Lambda, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model


INPUT_SHAPE = (93, 1029, 4)

def build_model():
    inputs = Input(shape=INPUT_SHAPE, name="input")
    conv1 = Conv2D(24, 5, strides=(2,16), activation='relu', name='conv1')(inputs)
    conv2 = Conv2D(36, 5, strides=(2,2), activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(48, 5, strides=(2,2), activation='relu', name='conv3')(conv2)
    deconv4 = Conv2DTranspose(36, 5, strides=(2,2), activation='relu', name='deconv4')(conv3)
    concat_deconv4 = concatenate([conv2, deconv4], name='concat_deconv4')

    deconv5a = Conv2DTranspose(24, 5, strides=(2,2), activation='relu', name='deconv5a')(concat_deconv4)
    concat_deconv5a = concatenate([conv1, deconv5a], name='concat_deconv5a')
    deconv6a = Conv2DTranspose(24, 5, strides=(2,16), activation='relu', name='deconv6a')(concat_deconv5a)

    deconv5b = Conv2DTranspose(24, 5, strides=(2,2), activation='relu', name='deconv5b')(concat_deconv4)
    concat_deconv5b = concatenate([conv1, deconv5b], name='concat_deconv5b')
    deconv6b = Conv2DTranspose(24, 5, strides=(2,16), activation='relu', name='deconv6b')(concat_deconv5b)

    model = Model(inputs=inputs, outputs=[deconv6a, deconv6b])
    print(model.summary())
    
    return model

def main():
    build_model()
    
if __name__ == '__main__':
    main()



