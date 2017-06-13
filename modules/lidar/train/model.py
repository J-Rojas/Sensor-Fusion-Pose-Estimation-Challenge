
# coding: utf-8

import os
import numpy as np
from numpy import random
import cv2
import keras
import json
from keras import backend as K
import tensorflow as tf
import globals

from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

# Disabling both USE_W1 and USE_W2 should result in typical categorical_cross_entropy loss
USE_W1 = True
USE_W2 = True


    
def custom_weighted_loss(input_shape, obj_to_bkg_ratio=0.00016, avg_obj_size=1000, loss_scaler=1000, weight_bb=1.0):

    def custom_loss(y_true, y_pred):
        #y_true = tf.Print(y_true, ['y_true:', tf.shape(y_true)])
        y_true_obj = y_true[:, :, 0:2]
        y_pred_obj = y_pred[:, :, 0:2]        
        y_true_bb = y_true[:, :, 2:]
        y_pred_bb = y_pred[:, :, 2:] 
        
        # the code here is only executed once, since these should all be graph operations. Do not expect Numpy
        # calculations or the like to work here, only Keras backend and tensor flow nodes.

        max_pixels = input_shape[0] * input_shape[1]

        log_softmax = tf.log(y_pred_obj, name="logsoftmax")        
        neglog_softmax = tf.scalar_mul(-1., log_softmax)

        pixel_loss = tf.multiply(y_true_obj, neglog_softmax, name="pixel_loss")       
        
        labels_bkg, labels_frg = tf.split(y_true_obj, 2, 2, name="split_2")
        bkg_frg_areas = tf.reduce_sum(y_true_obj, 1)
        bkg_area, frg_area = tf.split(bkg_frg_areas, 2, 1, name="split_1")

        # The branches here configure the graph differently. You can imagine these branches working as if the path
        # that was disabled didn't exist at all in the code. Each path should work independently.
        
        if USE_W1:
            w1_bkg_weights = tf.scalar_mul(obj_to_bkg_ratio, labels_bkg)
        else:
            w1_bkg_weights = labels_bkg
               
        frg_area_tiled = tf.tile(frg_area, tf.stack([1, max_pixels]))

        # prevent divide by zero, max is number of pixels
        frg_area_tiled = K.clip(frg_area_tiled, K.epsilon(), max_pixels)
        inv_frg_area = tf.div(tf.ones_like(frg_area_tiled), frg_area_tiled)

        w2_weights = tf.scalar_mul(avg_obj_size, inv_frg_area)
        w2_frg_weights = tf.multiply(labels_frg, tf.expand_dims(w2_weights, axis=2))                                  

        w1_times_w2 = tf.add(w1_bkg_weights, w2_frg_weights, name="w1_times_w2")
        weighted_loss = tf.multiply(w1_times_w2, pixel_loss, name="weighted_loss")        
        weighted_loss = tf.scalar_mul(loss_scaler, weighted_loss)

        loss_obj = tf.reduce_sum(weighted_loss, -1, name="loss")                          
        
        # weighted loss for regression branch
        #the following line needs to be changed to use prediction encoding
        diff_bb = tf.subtract(y_true_bb, y_pred_bb)       
        l2_norm = tf.norm(diff_bb)       
        
        weighted_loss_bb = tf.multiply(w2_frg_weights, l2_norm, name="weighted_l2_loss")
        loss_bb = tf.reduce_sum(weighted_loss_bb, -1, name="loss_bb")        
        loss_bb = tf.scalar_mul(weight_bb, loss_bb)
        #loss_bb = tf.Print(loss_bb, ["loss_bb", tf.shape(loss_bb), loss_bb])
        
        loss = tf.add(loss_obj, loss_bb, name="loss")
        #loss = tf.Print(loss, ["loss", tf.shape(loss), loss])
        # loss = K.categorical_crossentropy(softmax, y_true)

        return loss
    
    return custom_loss

def custom_weighted_cross_entropy(input_shape, obj_to_bkg_ratio=0.00016, avg_obj_size=1000, loss_scaler=1000):

    def custom_loss(y_true, y_pred):

        # the code here is only executed once, since these should all be graph operations. Do not expect Numpy
        # calculations or the like to work here, only Keras backend and tensor flow nodes.

        max_pixels = input_shape[0] * input_shape[1]

        log_softmax = tf.log(y_pred, name="logsoftmax")
        neglog_softmax = tf.scalar_mul(-1., log_softmax)

        pixel_loss = tf.multiply(y_true, neglog_softmax, name="pixel_loss")

        labels_bkg, labels_frg = tf.split(y_true, 2, 2, name="split_2")
        bkg_frg_areas = tf.reduce_sum(y_true, 1)
        bkg_area, frg_area = tf.split(bkg_frg_areas, 2, 1, name="split_1")

        # The branches here configure the graph differently. You can imagine these branches working as if the path
        # that was disabled didn't exist at all in the code. Each path should work independently.
        
        if USE_W1:
            w1_bkg_weights = tf.scalar_mul(obj_to_bkg_ratio, labels_bkg)
        else:
            w1_bkg_weights = labels_bkg

        if USE_W2:
            frg_area_tiled = tf.tile(frg_area, tf.stack([1, max_pixels]))

            # prevent divide by zero, max is number of pixels
            frg_area_tiled = K.clip(frg_area_tiled, K.epsilon(), max_pixels)
            inv_frg_area = tf.div(tf.ones_like(frg_area_tiled), frg_area_tiled)

            w2_weights = tf.scalar_mul(avg_obj_size, inv_frg_area)
            w2_frg_weights = tf.multiply(labels_frg, tf.expand_dims(w2_weights, axis=2))

        else:
            w2_frg_weights = labels_frg

        w1_times_w2 = tf.add(w1_bkg_weights, w2_frg_weights, name="w1_times_w2")
        weighted_loss = tf.multiply(w1_times_w2, pixel_loss, name="weighted_loss")
        weighted_loss = tf.scalar_mul(loss_scaler, weighted_loss)

        loss = tf.reduce_sum(weighted_loss, -1, name="loss")
        #loss = tf.Print(loss, ["loss", tf.shape(loss), loss])

        # loss = K.categorical_crossentropy(softmax, y_true)

        return loss
    
    return custom_loss

    
def build_model(input_shape, num_classes,
                use_regression=False,
                obj_to_bkg_ratio=0.00016,
                avg_obj_size=1000,
                weight_bb=1.0,
                metrics=None,
                trainable=True):

    # set channels last format
    K.set_image_data_format('channels_last')

    inputs = Input(shape=input_shape, name='input')
    flatten_input = Reshape((-1, input_shape[2]), name='flatten_input')(inputs)
    normalized = BatchNormalization(name='normalize', axis=1)(flatten_input)
    unflatten_input = Reshape((input_shape[0], input_shape[1], input_shape[2]), name='unflatten_input')(normalized)
    inputs_padded = ZeroPadding2D(padding=((0, 0), (0, 3)))(unflatten_input)
    conv1 = Conv2D(4, 5, strides=(2,4), activation='relu', name='conv1', padding='same',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(inputs_padded)
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
        deconv6a_flatten = Reshape((-1, num_classes), name='deconv6a_flatten')(deconv6a_crop)
        softmax = Activation('softmax',name='softmax')(deconv6a_flatten)
        classification_output = Lambda(lambda x: K.clip(x, K.epsilon(), 1), name='classification_output')(softmax) 
        
        deconv5b = Conv2DTranspose(24, 5, strides=(2,2), activation='relu', name='deconv5b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)
        deconv5b_padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(deconv5b)
        concat_deconv5b = concatenate([conv1, deconv5b_padded], name='concat_deconv5b')
        deconv6b = Conv2DTranspose(24, 5, strides=(2,4), activation='relu', name='deconv6b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5b)
        deconv6b_crop = Cropping2D(cropping=((3, 0), (0, 4)))(deconv6b)
        regression_output = Reshape((-1, 24), name='regression_output')(deconv6b_crop)
        
        # concatenate two outputs into one so that we can have one loss function       
        outputs = concatenate([classification_output, regression_output], name='outputs')        
        model = Model(inputs=inputs, outputs=outputs)
       
        model.compile(optimizer=Adam(lr=globals.LEARNING_RATE), 
                      loss=custom_weighted_loss(input_shape, obj_to_bkg_ratio, avg_obj_size, weight_bb),                            
                      metrics=metrics)                     
    else:
        flatten = Reshape((-1, num_classes), name='flatten')(deconv6a_crop)
        softmax = Activation('softmax',name='softmax')(flatten)
        softmax_clipped = Lambda(lambda x: K.clip(x, K.epsilon(), 1), name='clip_epsilon')(softmax)
        model = Model(inputs=inputs, outputs=softmax_clipped)
        model.compile(optimizer=Adam(lr=globals.LEARNING_RATE),
                      loss=custom_weighted_cross_entropy(input_shape, obj_to_bkg_ratio, avg_obj_size),
                      metrics=metrics)
        
    print(model.summary())    
    return model


def load_model(model_file, weights_file, input_shape, num_classes,
               obj_to_bkg_ratio=0.00016,
               avg_obj_size=1000,
               weight_bb=1.0,
               metrics=None):
    with open(model_file, 'r') as jfile:
        print("reading existing model and weights")
        model = keras.models.model_from_json(json.loads(jfile.read()))
        model.load_weights(weights_file)
        model.compile(optimizer=Adam(lr=globals.LEARNING_RATE),
                      loss=custom_weighted_cross_entropy(input_shape, obj_to_bkg_ratio, avg_obj_size),
                      metrics=metrics)

    return model


def test(model):
    # please change path if needed
    path = '../../dataset1/10/lidar_360/1490991699437114271_'
    if os.path.exists(path + 'height.png'):
        print('image found')
    height_img = cv2.imread(path + 'height.png') 
    height_gray = cv2.cvtColor(height_img, cv2.COLOR_RGB2GRAY)
    distance_img = cv2.imread(path + 'distance.png') 
    distance_gray = cv2.cvtColor(distance_img, cv2.COLOR_RGB2GRAY)
    intensity_img = cv2.imread(path + 'intensity.png') 
    intensity_gray = cv2.cvtColor(intensity_img, cv2.COLOR_RGB2GRAY)
    x = np.zeros(globals.INPUT_SHAPE)
    x[:, :, 0] = height_gray
    x[:, :, 1] = distance_gray
    x[:, :, 2] = intensity_gray

    label = np.zeros(globals.INPUT_SHAPE[:2])
    label[8:, 1242:1581] = 1  #bounding box of the obstacle vehicle
    y = to_categorical(label, num_classes=2) #1st dimension: off-vehicle, 2nd dimension: on-vehicle
    #print(np.nonzero(y[:,1])[0].shape[0])
    
    #place holder
    regression_label = np.zeros((1, globals.IMG_WIDTH*globals.IMG_HEIGHT, 24))    
    outputs = np.concatenate((np.asarray([y]), regression_label), axis=2)
    print(outputs.shape)
    model.fit(np.asarray([x]), outputs, batch_size=globals.BATCH_SIZE, epochs=globals.EPOCHS, verbose=1)


def main():
    model = build_model(globals.INPUT_SHAPE, 2, use_regression=True)
    test(model)
    
if __name__ == '__main__':
    main()



