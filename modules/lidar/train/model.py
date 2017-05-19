
# coding: utf-8

import os
import numpy as np
from numpy import random
import cv2
from keras import backend as K
import tensorflow as tf

from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam


BATCH_SIZE = 32
EPOCHS = 10

# Disabling both USE_W1 and USE_W2 should result in typical categorical_cross_entropy loss
USE_W1 = True
USE_W2 = True

def custom_weighted_cross_entropy(obj_to_bkg_ratio=0.00016, avg_obj_size=1000):

    def custom_loss(y_true, y_pred):
        softmax = tf.nn.softmax(y_pred, dim=2)

        # clip
        softmax = K.clip(softmax, K.epsilon(), 1)
        log_softmax = tf.log(softmax, name="logsoftmax")
        neglog_softmax = tf.scalar_mul(-1., log_softmax)

        pixel_loss = tf.multiply(y_true, neglog_softmax, name="pixel_loss")

        labels_bkg, labels_frg = tf.split(y_true, 2, 2, name="split_2")
        bkg_frg_areas = tf.reduce_sum(y_true, 1)
        bkg_area, frg_area = tf.split(bkg_frg_areas, 2, 1, name="split_1")

        if USE_W1:
            w1_bkg_weights = tf.scalar_mul(obj_to_bkg_ratio, labels_bkg)
        else:
            w1_bkg_weights = labels_bkg

        if USE_W2:
            frg_area_tiled = tf.tile(frg_area, tf.stack([1, 57632]))
            #frg_area_tiled = tf.Print(frg_area_tiled, ["frg_area_tiled", tf.shape(frg_area_tiled), frg_area_tiled])

            frg_area_tiled = K.clip(frg_area_tiled, K.epsilon(), 1)
            inv_frg_area = tf.div(tf.ones_like(frg_area_tiled), frg_area_tiled)
            #inv_frg_area = tf.Print(inv_frg_area, ["inv_frg_area", tf.shape(inv_frg_area), inv_frg_area])

            w2_weights = tf.scalar_mul(avg_obj_size, inv_frg_area)
            w2_frg_weights = tf.multiply(labels_frg, tf.expand_dims(w2_weights, axis=2))
        else:
            w2_frg_weights = labels_frg

        w1_times_w2 = tf.add(w1_bkg_weights, w2_frg_weights, name="w1_times_w2")
        weighted_loss = tf.multiply(w1_times_w2, pixel_loss, name="weighted_loss")

        loss = tf.reduce_sum(weighted_loss, -1, name="loss")
        loss = tf.Print(loss, ["loss", tf.shape(loss), loss])

        # XXX temporarily disable custom weighted categorical cross entropy for validation evaluation
        # loss = K.categorical_crossentropy(softmax, y_true)

        return loss
    
    return custom_loss


def build_model(input_shape, num_classes,
                use_regression=False,
                obj_to_bkg_ratio=0.00016,
                avg_obj_size=1000,
                metrics=None):

    # set channels last format
    K.set_image_data_format('channels_last')

    inputs = Input(shape=input_shape, name='input')
    inputs_padded = ZeroPadding2D(padding=((0, 0), (0, 3)))(inputs)
    normalized = BatchNormalization(name='normalize')(inputs_padded)
    conv1 = Conv2D(4, 5, strides=(2,4), activation='relu', name='conv1', padding='same',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(normalized)
    conv2 = Conv2D(6, 5, strides=(2,2), activation='relu', name='conv2',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(conv1)
    conv3 = Conv2D(12, 5, strides=(2,2), activation='relu', name='conv3',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(conv2)
    deconv4 = Conv2DTranspose(16, 5, strides=(2,2), activation='relu', name='deconv4',
                              kernel_initializer='random_uniform', bias_initializer='zeros')(conv3)
    deconv4_padded = ZeroPadding2D(padding=((1, 0), (0, 1)))(deconv4)
    concat_deconv4 = concatenate([conv2, deconv4_padded], name='concat_deconv4')

    # classification task
    deconv5a = Conv2DTranspose(8, 5, strides=(2,2), activation='relu', name='deconv5a',
                               kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)
    deconv5a_padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(deconv5a)
    concat_deconv5a = concatenate([conv1, deconv5a_padded], name='concat_deconv5a')
    deconv6a = Conv2DTranspose(2, 5, strides=(2,4), name='deconv6a', padding='same',
                               kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5a)
    deconv6a_crop = Cropping2D(cropping=((0, 0), (0, 3)))(deconv6a)

    # regression task
    if use_regression:
        deconv5b = Conv2DTranspose(24, 5, strides=(2,2), activation='relu', name='deconv5b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)
        concat_deconv5b = concatenate([conv1, deconv5b], name='concat_deconv5b')
        deconv6b = Conv2DTranspose(24, 5, strides=(2,4), activation='relu', name='deconv6b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5b)

        # TODO: the output layer may need reshaping
        model = Model(inputs=inputs, outputs=[deconv6a, deconv6b])
        model.compile(optimizer=Adam(lr=0.001), 
                      loss={'deconv6a': 'categorical_crossentropy', 'deconv6b': 'mse'}, 
                      metrics={'deconv6a': 'accuracy', 'deconv6b': 'mse'})
    else:
        flatten = Reshape((-1, num_classes), name='flatten')(deconv6a_crop)
        model = Model(inputs=inputs, outputs=flatten)
        model.compile(optimizer=Adam(lr=0.001),
                      loss=custom_weighted_cross_entropy(obj_to_bkg_ratio, avg_obj_size),
                      metrics=metrics)
        
    print(model.summary())    
    return model


def test(model):
    # please change path if needed
    path = '../../sample/9_f/lidar_360/1491843463372931936_'
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



