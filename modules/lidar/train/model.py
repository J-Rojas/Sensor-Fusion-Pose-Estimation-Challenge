
# coding: utf-8

import os
import numpy as np
from numpy import random
import cv2
import keras
import json
from keras import backend as K
import tensorflow as tf
from globals import NUM_CHANNELS, NUM_CLASSES, NUM_REGRESSION_OUTPUTS, LEARNING_RATE
import sys
sys.path.append('../')
from process.globals import X_MIN, Y_MIN, RES_RAD

from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

# Disabling both USE_W1 and USE_W2 should result in typical categorical_cross_entropy loss
USE_W1 = True
USE_W2 = True
    
def custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio=0.00016, avg_obj_size=1000, loss_scaler=1000):

    def custom_loss(y_true, y_pred):
        # the code here is only executed once, since these should all be graph operations. Do not expect Numpy
        # calculations or the like to work here, only Keras backend and tensor flow nodes.

        max_pixels = input_shape[0] * input_shape[1]
        #y_pred = tf.Print(y_pred, ["y_pred", tf.shape(y_pred)])
        
        if use_regression:                  
            input, y_true_obj, y_true_bb = tf.split(y_true, [NUM_CHANNELS, NUM_CLASSES, NUM_REGRESSION_OUTPUTS], 2)          
            _, y_pred_obj, y_pred_bb = tf.split(y_pred, [NUM_CHANNELS, NUM_CLASSES, NUM_REGRESSION_OUTPUTS], 2)
                        
            distance, height, _ = tf.split(input, 3, 2) 
            distance = tf.reshape(distance, [-1, max_pixels])
            height = tf.reshape(height, [-1, max_pixels])           
            distance_img = tf.reshape(distance, input_shape[:2])
            
            img_x = tf.cast(tf.range(input_shape[1]), tf.float32)             
            theta = tf.scalar_mul(RES_RAD[1], tf.add(img_x,  tf.fill(tf.shape(img_x), X_MIN)))            
            theta = tf.tile(theta, [input_shape[0]])            
            theta = tf.tile(theta, [tf.shape(distance)[0]])            
            theta = tf.reshape(theta, [-1, max_pixels])
            
            img_y = tf.cast(tf.range(input_shape[0]), tf.float32) 
            phi = tf.scalar_mul(RES_RAD[0], tf.add(img_y,  tf.fill(tf.shape(img_y), Y_MIN)))
            phi = tf.tile(phi, [input_shape[1]])                                      
            phi = tf.tile(phi, [tf.shape(distance)[0]])           
            phi = tf.reshape(phi, [-1, max_pixels])
            
            px = tf.multiply(distance, tf.cos(theta))
            py = - tf.multiply(distance, tf.sin(theta))                       
            p = tf.stack([px, py, height])
            p = tf.reshape(p, (-1, max_pixels, 3))            
            
            #rotation around z axis
            theta_zeros = tf.fill(tf.shape(theta), 0.0)
            theta_ones = tf.fill(tf.shape(theta), 1.0)                                       
            rot_z = tf.stack([tf.cos(theta), -tf.sin(theta), theta_zeros, 
                              tf.sin(theta), tf.cos(theta),  theta_zeros,
                              theta_zeros,   theta_zeros,    theta_ones])                        
            rot_z = tf.reshape(rot_z, (-1, max_pixels, 3, 3)) 
            
            #rotation around y axis  
            phi_zeros = tf.fill(tf.shape(phi), 0.0)
            phi_ones = tf.fill(tf.shape(phi), 1.0)                
            rot_y = tf.stack([tf.cos(phi), phi_zeros, tf.sin(phi),
                              phi_zeros,   phi_ones,  phi_zeros,
                             -tf.sin(phi), phi_zeros, tf.cos(phi)])
            rot_y = tf.reshape(rot_y, (-1, max_pixels, 3, 3))
            
            rot = tf.matmul(rot_z, rot_y)
            rot_T = tf.matrix_transpose(rot)
            rot_T = tf.reshape(rot_T, (-1, max_pixels, 3, 3))
            
            #8 corners of the bounding box
            c1, c2, c3, c4, c5, c6, c7, c8 = tf.split(y_pred_bb, 8, 2)           
            c1_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c1, p), 3))            
            c2_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c2, p), 3))            
            c3_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c3, p), 3))           
            c4_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c4, p), 3))           
            c5_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c5, p), 3))           
            c6_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c6, p), 3))           
            c7_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c7, p), 3))           
            c8_prime = tf.matmul(rot_T, tf.expand_dims(tf.subtract(c8, p), 3))           

            y_pred_bb_encoded = tf.stack([c1_prime, c2_prime, c3_prime, c4_prime,
                                          c5_prime, c6_prime, c7_prime, c8_prime], 2)
            y_pred_bb_encoded = tf.reshape(y_pred_bb_encoded, (-1, max_pixels, NUM_REGRESSION_OUTPUTS))                                                
            #y_pred_bb_encoded = tf.Print(y_pred_bb_encoded, ["y_pred_bb_encoded nonzero:", tf.count_nonzero(y_pred_bb_encoded), " max:", tf.reduce_max(y_pred_bb_encoded), " min:", tf.reduce_mean(y_pred_bb_encoded)])            
        else:
            y_true_obj = y_true
            y_pred_obj = y_pred                              

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

        loss = loss_obj = tf.reduce_sum(weighted_loss, -1, name="loss")                          
        
        # weighted loss for regression branch
        #the following line needs to be changed to use prediction encoding
        if use_regression:        
            diff_bb = tf.subtract(y_true_bb, y_pred_bb_encoded)    
            l2_norm = tf.norm(diff_bb)       
            
            weighted_loss_bb = tf.multiply(w2_frg_weights, l2_norm, name="weighted_l2_loss")
            loss_bb = tf.reduce_sum(weighted_loss_bb, -1, name="loss_bb")        
            loss_bb = tf.scalar_mul(weight_bb, loss_bb)
            #loss_obj = tf.Print(loss_obj, ["loss_obj max", tf.reduce_max(loss_obj), " mean:", tf.reduce_mean(loss_obj)])
            #loss_bb = tf.Print(loss_bb, ["loss_bb max:", tf.reduce_max(loss_bb), " mean:", tf.reduce_mean(loss_bb)])            
            loss = tf.add(loss_obj, loss_bb, name="loss")
            
        #loss = tf.Print(loss, ["loss", tf.shape(loss), loss])       

        return loss
    
    return custom_loss
    
def build_model(input_shape, num_classes,
                use_regression=True,
                obj_to_bkg_ratio=0.00016,
                avg_obj_size=1000,
                weight_bb=1e-5,
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
    deconv6a_flatten = Reshape((-1, num_classes), name='deconv6a_flatten')(deconv6a_crop)
    softmax = Activation('softmax',name='softmax')(deconv6a_flatten)
    output = classification_output = Lambda(lambda x: K.clip(x, K.epsilon(), 1), name='classification_output')(softmax)

    # regression task
    if use_regression:        
        deconv5b = Conv2DTranspose(NUM_REGRESSION_OUTPUTS, 5, strides=(2,2), activation='relu', name='deconv5b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv4)
        deconv5b_padded = ZeroPadding2D(padding=((1, 0), (0, 0)))(deconv5b)
        concat_deconv5b = concatenate([conv1, deconv5b_padded], name='concat_deconv5b')
        deconv6b = Conv2DTranspose(NUM_REGRESSION_OUTPUTS, 5, strides=(2,4), activation='relu', name='deconv6b',
                                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_deconv5b)
        deconv6b_crop = Cropping2D(cropping=((3, 0), (0, 4)))(deconv6b)
        regression_output = Reshape((-1, NUM_REGRESSION_OUTPUTS), name='regression_output')(deconv6b_crop)
        
        # concatenate two outputs into one so that we can have one loss function       
        output = concatenate([flatten_input, classification_output, regression_output], name='outputs')        
 
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio, avg_obj_size),
                  metrics=metrics)
                      
    print(model.summary())    
    return model


def load_model(model_file, weights_file, input_shape, num_classes, use_regression,
               obj_to_bkg_ratio=0.00016,
               avg_obj_size=1000,
               weight_bb=1e-5,
               metrics=None):
    with open(model_file, 'r') as jfile:
        print("reading existing model and weights")
        model = keras.models.model_from_json(json.loads(jfile.read()))
        model.load_weights(weights_file)
        model.compile(optimizer=Adam(lr=LEARNING_RATE),
                      loss=custom_weighted_loss(input_shape, use_regression, weight_bb, obj_to_bkg_ratio, avg_obj_size),
                      metrics=metrics)

    return model


def test(model):
    # please change path if needed
    path = '../../data/10/lidar_360/1490991699437114271_'
    if os.path.exists(path + 'height.png'):
        print('image found')
    height_img = cv2.imread(path + 'height.png') 
    height_gray = cv2.cvtColor(height_img, cv2.COLOR_RGB2GRAY)
    distance_img = cv2.imread(path + 'distance.png') 
    distance_gray = cv2.cvtColor(distance_img, cv2.COLOR_RGB2GRAY)
    intensity_img = cv2.imread(path + 'intensity.png') 
    intensity_gray = cv2.cvtColor(intensity_img, cv2.COLOR_RGB2GRAY)
    x = np.zeros(INPUT_SHAPE)
    x[:, :, 0] = height_gray
    x[:, :, 1] = distance_gray
    x[:, :, 2] = intensity_gray   
    x_reshaped = np.asarray([x]).reshape((1, -1, NUM_CHANNELS))    
    
    label = np.zeros(INPUT_SHAPE[:2])
    label[8:, 1242:1581] = 1  #bounding box of the obstacle vehicle
    y = to_categorical(label, num_classes=NUM_CLASSES) #1st dimension: off-vehicle, 2nd dimension: on-vehicle
    #print(np.nonzero(y[:,1])[0].shape[0])
    
    #place holder
    regression_label = np.zeros((1, IMG_WIDTH*IMG_HEIGHT, NUM_REGRESSION_OUTPUTS))        
    outputs = np.concatenate((x_reshaped, np.asarray([y]), regression_label), axis=2)
    #outputs = np.asarray([y])
    print(outputs.shape)
    model.fit(np.asarray([x]), outputs, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)


def main():
    model = build_model(INPUT_SHAPE, NUM_CLASSES, use_regression=True)
    test(model)
    
if __name__ == '__main__':
    main()



