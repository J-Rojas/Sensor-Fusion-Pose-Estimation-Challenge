
# coding: utf-8

import os
import numpy as np
from numpy import random
import cv2

from keras.layers import Input, concatenate, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

INPUT_SHAPE = (93, 1029, 2)
BATCH_SIZE = 32
EPOCHS = 10

def build_model(use_regression=False):
    inputs = Input(shape=INPUT_SHAPE, name="input")
    conv1 = Conv2D(4, 5, strides=(2,4), activation='relu', name='conv1')(inputs)
    conv2 = Conv2D(6, 5, strides=(2,2), activation='relu', name='conv2')(conv1)
    conv3 = Conv2D(12, 5, strides=(2,2), activation='relu', name='conv3')(conv2)
    deconv4 = Conv2DTranspose(16, 5, strides=(2,2), activation='relu', name='deconv4')(conv3)
    concat_deconv4 = concatenate([conv2, deconv4], name='concat_deconv4')

    #classification task
    deconv5a = Conv2DTranspose(8, 5, strides=(2,2), activation='relu', name='deconv5a')(concat_deconv4)
    concat_deconv5a = concatenate([conv1, deconv5a], name='concat_deconv5a')
    deconv6a = Conv2DTranspose(2, 5, strides=(2,4), activation='relu', name='deconv6a')(concat_deconv5a)

    #regression task
    if use_regression:
        deconv5b = Conv2DTranspose(24, 5, strides=(2,2), activation='relu', name='deconv5b')(concat_deconv4)
        concat_deconv5b = concatenate([conv1, deconv5b], name='concat_deconv5b')
        deconv6b = Conv2DTranspose(24, 5, strides=(2,4), activation='relu', name='deconv6b')(concat_deconv5b)

        #TODO: the output layer may need reshaping
        model = Model(inputs=inputs, outputs=[deconv6a, deconv6b])
        model.compile(optimizer=Adam(lr=0.001), 
                      loss={'deconv6a': 'categorical_crossentropy', 'deconv6b': 'mse'}, 
                      metrics={'deconv6a': 'accuracy', 'deconv6b': 'mse'})
    else:
        flatten = Reshape((-1, 2), name='flatten')(deconv6a)
        model = Model(inputs=inputs, outputs=flatten)   
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        
    print(model.summary())    
    return model

def test(model):
    path = 'sample/9_f/lidar_360/1491843463372931936_'
    height_img = cv2.imread(path + 'height.png') 
    height_gray = cv2.cvtColor(height_img, cv2.COLOR_RGB2GRAY)
    distance_img = cv2.imread(path + 'distance.png') 
    distance_gray = cv2.cvtColor(distance_img, cv2.COLOR_RGB2GRAY)
    x = np.zeros((93, 1029, 2))
    x[:, :, 0] = height_gray
    x[:, :, 1] = distance_gray

    label = np.ones((93, 1029))
    label[27:, 621:826] = 0  #bounding box of the obstacle vehicle
    y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle
    model.fit(np.asarray([x]), np.asarray([y]), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)


def main():
    model = build_model()
    test(model)
    
if __name__ == '__main__':
    main()



