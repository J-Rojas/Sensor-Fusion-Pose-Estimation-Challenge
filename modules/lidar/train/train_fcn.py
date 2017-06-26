import sys
sys.path.append('../')
import argparse
import datetime
import json
import os
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import globals
from globals import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, \
                    NUM_CHANNELS, NUM_CLASSES, EPOCHS, INPUT_SHAPE, \
                    K_NEGATIVE_SAMPLE_RATIO_WEIGHT, LEARNING_RATE, \
                    IMG_CAM_WIDTH, IMG_CAM_HEIGHT, NUM_CAM_CHANNELS, INPUT_SHAPE_CAM

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Input, concatenate, Reshape, BatchNormalization, Activation, Lambda, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from common.camera_model import CameraModel
from process.globals import CAM_IMG_BOTTOM, CAM_IMG_TOP
from loader import get_data_and_ground_truth, data_generator_train, \
                   data_number_of_batches_per_epoch, filter_camera_data_and_gt, \
                   generate_index_list, file_prefix_for_timestamp, \
                   load_data
from model import build_model, load_model
from pretrain import calculate_population_weights
from common import pr_curve_plotter
from common.csv_utils import foreach_dirset
from train import LossHistory

def load_gt(indicies, centroid_rotation, gt):
    
    batch_index = 0

    for ind in indicies:
        gt[batch_index,0] = centroid_rotation[0][ind]
        gt[batch_index,1] = centroid_rotation[1][ind]
        gt[batch_index,2] = centroid_rotation[2][ind]
        gt[batch_index,3] = centroid_rotation[3][ind]
        gt[batch_index,4] = centroid_rotation[4][ind]
        gt[batch_index,5] = centroid_rotation[5][ind]
        batch_index += 1

def data_generator_FCN(obs_centroids, obs_size, 
                        pickle_dir_and_prefix_cam, pickle_dir_and_prefix_lidar,
                        batch_size):
                        
    tx = obs_centroids[0]
    ty = obs_centroids[1]
    tz = obs_centroids[2]
    rx = obs_centroids[3]
    ry = obs_centroids[4]
    rz = obs_centroids[5]
    
    obsl = obs_size[0]
    obsw = obs_size[1]
    obsh = obs_size[2]

    cam_images = np.ndarray(shape=(batch_size, globals.IMG_CAM_HEIGHT, 
                    globals.IMG_CAM_WIDTH, globals.NUM_CAM_CHANNELS), dtype=float)    
    lidar_images = np.ndarray(shape=(batch_size, globals.IMG_HEIGHT, 
                    globals.IMG_WIDTH, globals.NUM_CHANNELS), dtype=float)
    centroid_rotation = np.ndarray(shape=(batch_size, 6), dtype=float)

    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix_cam, batch_size)

    indicies_list = np.arange(len(tx))

    while 1:            

        indicies = generate_index_list(indicies_list, False, num_batches, batch_size)

        for batch in range(num_batches):

            batch_indicies = indicies[batch * batch_size:batch * batch_size + batch_size]

            load_data(batch_indicies, lidar_images, pickle_dir_and_prefix_lidar, "lidar", globals.NUM_CHANNELS)
            load_data(batch_indicies, cam_images, pickle_dir_and_prefix_cam, "camera", globals.NUM_CAM_CHANNELS)
            load_gt(batch_indicies, obs_centroids, centroid_rotation)
 
            yield ([cam_images, lidar_images], centroid_rotation)
   
    
def get_data_and_ground_truth_matching_lidar_cam_frames(csv_sources, parent_dir):
    
    txl_cam = []
    tyl_cam = []
    tzl_cam = []
    rxl_cam = []
    ryl_cam = []
    rzl_cam = []
    timestamps_cam = []
    obsl_cam = []
    obsw_cam = []
    obsh_cam = []
    
    pickle_dir_and_prefix_cam = []
    pickle_dir_and_prefix_lidar = []

    def process(dirset):

        lidar_truth_fname = dirset.dir+"/obs_poses_interp_transform.csv"
        cam_truth_fname = dirset.dir+"/obs_poses_camera.csv"

        df_lidar_truths = pd.read_csv(lidar_truth_fname)
        lidar_truths_list = df_lidar_truths['timestamp'].tolist()
         
        #print lidar_rows[:,'timestamp']   
        def nearest_lidar_timestamp(cam_ts):
            x = min(lidar_truths_list, key=lambda x:abs(x-cam_ts))
            return x
            

        with open(cam_truth_fname) as csvfile_2:
            readCSV_2 = csv.DictReader(csvfile_2, delimiter=',')

            for row2 in readCSV_2:
                ts = row2['timestamp']
                tx = row2['tx']
                ty = row2['ty']
                tz = row2['tz']
                rx = row2['rx']
                ry = row2['ry']
                rz = row2['rz']

                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "camera", ts)
                pickle_dir_and_prefix_cam.append(pickle_dir_prefix)
                txl_cam.append(float(tx))
                tyl_cam.append(float(ty))
                tzl_cam.append(float(tz))
                rxl_cam.append(float(rx))
                ryl_cam.append(float(ry))
                rzl_cam.append(float(rz))
                timestamps_cam.append(ts)
                obsl_cam.append(float(dirset.mdr['l']))
                obsw_cam.append(float(dirset.mdr['w']))
                obsh_cam.append(float(dirset.mdr['h']))
                lidar_ts = nearest_lidar_timestamp(int(ts))
                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, "lidar", str(lidar_ts))
                pickle_dir_and_prefix_lidar.append(pickle_dir_prefix)
                
                
                
    foreach_dirset(csv_sources, parent_dir, process)

    obs_centroid = [txl_cam, tyl_cam, tzl_cam, rxl_cam, ryl_cam, rzl_cam, timestamps_cam]
    obs_size = [obsl_cam, obsw_cam, obsh_cam]
      
    
    return obs_centroid, pickle_dir_and_prefix_cam, obs_size, pickle_dir_and_prefix_lidar


def build_FCN(input_layer, output_layer, net_name, metrics=None, trainable=True):

#    print input_layer.shape
#    labels_bkg, labels_frg = tf.split(output_layer, 2, 2, name='split_'+net_name)
#    reshaped_out = tf.reshape(labels_frg, input_layer.shape, name='reshaped_'+net_name)

#    pool_out = MaxPooling2D()(output_layer)
    flatten_out = Flatten()(output_layer)
    dense_1 = Dense(32, activation='relu', name='dense1_'+net_name, use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(flatten_out)
    dense_2 = Dense(64, activation='relu', name='dense2_'+net_name, use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense_1)
    dense_3 = Dense(64, activation='relu', name='dense3_'+net_name, use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense_2)                  
    return dense_3

def build_FCN_cam_lidar(cam_inp, lidar_inp, cam_net_out, lidar_net_out, metrics=None, trainable=True):

    cam_net_out = build_FCN(cam_inp, cam_net_out, "cam")
    lidar_net_out = build_FCN(lidar_inp, lidar_net_out, "lidar")
    concat_input = concatenate([cam_net_out, lidar_net_out])
    
    dense_1 = Dense(32, activation='relu', name='dense1', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_input)
    dropout_1 = Dropout(0.3)(dense_1)
    dense_2 = Dense(64, activation='relu', name='dense2', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dropout_1)
    dropout_2 = Dropout(0.3)(dense_2)
    dense_3 = Dense(6, activation='relu', name='dense3', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dropout_2)

    model = Model(inputs=[cam_inp, lidar_inp], outputs=dense_3)                   
    model.compile(optimizer=Adam(),
                  loss="mean_squared_error", metrics=['mae'])
    

    print(model.summary())
    return model

def build_FCN_2(cam_inp, lidar_inp, cam_net_out, lidar_net_out, metrics=None, trainable=True):

    cam_inp = Input(shape=(BATCH_SIZE,6), name='input_cam')
    lidar_inp = Input(shape=(BATCH_SIZE,6), name='input_lidar')
    
    concat_input = concatenate([cam_inp, lidar_inp])
    
    dense_1 = Dense(32, activation='relu', name='dense1', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(concat_input)
    dense_2 = Dense(64, activation='relu', name='dense2', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense_1)
    dense_3 = Dense(6, activation='relu', name='dense3', use_bias=True,
                   kernel_initializer='random_uniform', bias_initializer='zeros')(dense_2)

    model = Model(inputs=[cam_inp, lidar_inp], outputs=dense_3)                   
    model.compile(optimizer=Adam(),
                  loss="mean_squared_error", metrics=['mae'])
    

    print(model.summary())
    return model
    
def main():

    parser = argparse.ArgumentParser(description='FCN trainer for radar/camera/lidar')
    parser.add_argument("--train_file", type=str, default="../data/train_folders.csv",
                        help="list of data folders for training")
    parser.add_argument("--val_file", type=str, default="../data/validation_folders.csv",
                        help="list of data folders for validation")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--camera_model', type=str, default="", help='Model Filename')
    parser.add_argument('--camera_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--lidar_model', type=str, default="", help='Model Filename')
    parser.add_argument('--lidar_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--fcn_model', type=str, default="", help='Model Filename')
    parser.add_argument('--fcn_weights', type=str, default="", help='Weights Filename')
    parser.add_argument('--outdir', type=str, default="./", help='output directory')
    parser.add_argument('--camera_calibration_model', type=str, help='Camera calibration yaml')
    parser.add_argument('--lidar2cam_model', type=str, help='Lidar to Camera calibration yaml')
    parser.add_argument('--cache', type=str, default=None, help='Cache data')

    args = parser.parse_args()
    train_file = args.train_file
    validation_file = args.val_file
    outdir = args.outdir
    dir_prefix = args.dir_prefix
    camera_model = CameraModel()
    camera_model.load_camera_calibration(args.camera_calibration_model, args.lidar2cam_model)

    skip_frames_indexes = []
       
    cache_train, cache_val = None, None
    if args.cache is not None:
        cache_train = {'data': None, 'labels': None}
        cache_val = {'data': None, 'labels': None}


    if args.camera_model == "" or args.camera_weights == "":
        print "need to enter camera model/weights file"
        exit(1)

    if args.lidar_model == "" or args.lidar_weights == "":
        print "need to enter lidar model/weights file"
        exit(1)
        
        
                        
    camera_net = load_model(args.camera_model, args.camera_weights,
                       INPUT_SHAPE_CAM, NUM_CLASSES, trainable=False)
    cam_inp_layer = camera_net.input
    cam_out_layer = camera_net.output

    lidar_net = load_model(args.lidar_model, args.lidar_weights,
                       INPUT_SHAPE, NUM_CLASSES, trainable=False)
    lidar_inp_layer = lidar_net.input
    lidar_out_layer = lidar_net.output
    
    cam_lidar_net = build_FCN_cam_lidar(cam_inp_layer, lidar_inp_layer, 
                        cam_out_layer, lidar_out_layer)

    # save the model
    with open(os.path.join(outdir, 'fcn_model.json'), 'w') as outfile:
            json.dump(cam_lidar_net.to_json(), outfile)   
                                
    train_data = get_data_and_ground_truth_matching_lidar_cam_frames(train_file, dir_prefix)
    val_data = get_data_and_ground_truth_matching_lidar_cam_frames(validation_file, dir_prefix)

    # number of batches per epoch
    n_batches_per_epoch_train = data_number_of_batches_per_epoch(train_data[1], BATCH_SIZE)
    n_batches_per_epoch_val = data_number_of_batches_per_epoch(val_data[1], BATCH_SIZE)

    print("Number of batches per epoch: {}".format(n_batches_per_epoch_train))
    print("start time:")
    print(datetime.datetime.now())

    checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'fcn_weights.{epoch:02d}-{loss:.4f}.hdf5'),
                                   verbose=1, save_weights_only=True)
    tensorboard = TensorBoard(histogram_freq=1, log_dir=os.path.join(outdir, 'tensorboard/'),
                              write_graph=True, write_images=False)
    loss_history = LossHistory()

    try:

        cam_lidar_net.fit_generator(
            data_generator_FCN(
                train_data[0], train_data[2], train_data[1], 
                train_data[3], BATCH_SIZE
            ),  # generator
            n_batches_per_epoch_train,  # number of batches per epoch
            validation_data=data_generator_FCN(
                val_data[0], val_data[2], val_data[1], 
                val_data[3], BATCH_SIZE
            ),
            validation_steps=n_batches_per_epoch_val,  # number of batches per epoch
            epochs=EPOCHS,
            callbacks=[checkpointer, tensorboard, loss_history],
            verbose=1
        )
        
    except KeyboardInterrupt:
        print('\n\nExiting training...')

    print("stop time:")
    print(datetime.datetime.now())
    # save model weights
    cam_lidar_net.save_weights(os.path.join(outdir, "fcn_model.h5"), True)

    #print precision_recall_array
    pr_curve_plotter.plot_pr_curve(loss_history, outdir)


if __name__ == '__main__':
    main()
