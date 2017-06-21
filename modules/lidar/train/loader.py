import sys
sys.path.append('../')
import os
import numpy as np
import glob
import argparse
import csv
import sys
import pickle
import pandas as pd
import globals
from common.csv_utils import foreach_dirset
from random import randrange
from collections import defaultdict
from encoder import generate_label, get_label_bounds


def usage():
    print('Loads training data with ground truths and generate training batches')
    print('Usage: python loader.py --input_csv_file [csv file of data folders]')


def data_number_of_batches_per_epoch(data, BATCH_SIZE):
    size = len(data)
    return int(size / BATCH_SIZE) + (1 if size % BATCH_SIZE != 0 else 0)

#
# rotate images/labels randomly
#
def data_random_rotate(image, label, obj_center, obj_size):
    
    # get bounding box of object in 2D
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = \
        get_label_bounds(obj_center[0], obj_center[1], obj_center[2], obj_size[0], obj_size[1], obj_size[2])
    
    # do not rotate if object rolls partially to right/left of the image  
    # get another random number
    rotate_by = randrange(0, globals.IMG_WIDTH)
    while upper_left_x+rotate_by <= globals.IMG_WIDTH <= lower_right_x+rotate_by:
        rotate_by = randrange(0, globals.IMG_WIDTH)
    
    #print "rotate_by: " + str(rotate_by)
    label_reshaped = np.reshape(label, (globals.IMG_HEIGHT, globals.IMG_WIDTH, \
                                globals.NUM_CLASSES+globals.NUM_REGRESSION_OUTPUTS))
    rotated_label = np.roll(label_reshaped, rotate_by, axis=1)
    rotated_flatten_label = np.reshape(rotated_label, (globals.IMG_HEIGHT*globals.IMG_WIDTH, \
                                       globals.NUM_CLASSES+globals.NUM_REGRESSION_OUTPUTS))
    rotated_img = np.roll(image, rotate_by, axis=1)
    
    # copy back rotated parts to original images/label
    np.copyto(image, rotated_img)
    np.copyto(label, rotated_flatten_label)

# 
# rotate data in a given batch
#
def batch_random_rotate(indicies, images, labels, tx, ty, tz, obsl, obsw, obsh):

    img_ind = 0
    for ind in indicies:

        obj_center = [tx[ind], ty[ind], tz[ind]]
        obj_size = [obsl[ind], obsw[ind], obsh[ind]]
        data_random_rotate(images[img_ind], labels[img_ind], obj_center, obj_size)

        img_ind += 1


def generate_index_list(indicies_list, randomize, num_batches, batch_size):
    if randomize:
        np.random.shuffle(indicies_list)

    indicies = indicies_list
    if len(indicies_list) < num_batches * batch_size:
        # add records from entire set to fill remaining space in batch
        indicies_list_rem = np.arange(len(indicies_list))
        if randomize:
            np.random.shuffle(indicies_list_rem)
        rem = num_batches * batch_size - len(indicies_list)
        indicies = np.concatenate((indicies_list, indicies_list_rem[0:rem]))

    return indicies

#
# read in images/ground truths batch by batch
#
def data_generator_train(obs_centroids, obs_size, pickle_dir_and_prefix, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, randomize=True, augment=True):
    tx = obs_centroids[0]
    ty = obs_centroids[1]
    tz = obs_centroids[2]
    obsl = obs_size[0]
    obsw = obs_size[1]
    obsh = obs_size[2]

    images = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    obj_labels = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT*IMG_WIDTH, NUM_CLASSES + globals.NUM_REGRESSION_OUTPUTS), dtype=np.uint8)    

    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix, BATCH_SIZE)

    indicies_list = np.arange(len(tx))

    while 1:

        indicies = generate_index_list(indicies_list, randomize, num_batches, BATCH_SIZE)

        for batch in range(num_batches):

            batch_indicies = indicies[batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE]

            load_lidar_data(batch_indicies, images, pickle_dir_and_prefix)
            load_label_data(batch_indicies, obj_labels, tx, ty, tz, obsl, obsw, obsh,
                            (IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES))
            if augment:
                batch_random_rotate(batch_indicies, images, obj_labels, tx, ty, tz, obsl, obsw, obsh)
            
            images_flattened = np.reshape(images, (-1, IMG_HEIGHT*IMG_WIDTH, NUM_CHANNELS))
            obj_labels_appended = np.concatenate((images_flattened, obj_labels), 2)                        
            
            yield (images, obj_labels_appended)


def data_generator_predict(pickle_dir_and_prefix, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES):

    images = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix, BATCH_SIZE)

    indicies_list = np.arange(len(pickle_dir_and_prefix))

    while 1:

        indicies = generate_index_list(indicies_list, True, num_batches, BATCH_SIZE)

        for batch in range(num_batches):

            batch_indicies = indicies[batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE]

            load_lidar_data(batch_indicies, images, pickle_dir_and_prefix)
            yield images


def load_lidar_data(indicies, images, pickle_dir_and_prefix):

    batch_index = 0

    for ind in indicies:

        fname = pickle_dir_and_prefix[ind] + "_distance_float.lidar.p"

        f = open(fname, 'rb')
        pickle_data = pickle.load(f)
        img_arr = np.asarray(pickle_data, dtype='float32')
        np.copyto(images[batch_index, :, :, 0], img_arr)
        f.close();

        fname = pickle_dir_and_prefix[ind] + "_height_float.lidar.p"
        f = open(fname, 'rb')
        pickle_data = pickle.load(f)
        img_arr = np.asarray(pickle_data, dtype='float32')
        np.copyto(images[batch_index, :, :, 1], img_arr)
        f.close();

        fname = pickle_dir_and_prefix[ind] + "_intensity_float.lidar.p"
        f = open(fname, 'rb')
        pickle_data = pickle.load(f)
        img_arr = np.asarray(pickle_data, dtype='float32')
        np.copyto(images[batch_index, :, :, 2], img_arr)
        f.close()

        batch_index += 1


def load_label_data(indicies, obj_labels, tx, ty, tz, obsl, obsw, obsh, shape):

    batch_index = 0

    for ind in indicies:

        label = generate_label(tx[ind], ty[ind], tz[ind], obsl[ind], obsw[ind], obsh[ind], shape)
        # label = np.ones(shape=(IMG_HEIGHT, IMG_WIDTH),dtype=np.dtype('u2'))
        np.copyto(obj_labels[batch_index], np.uint8(label))        
        batch_index += 1


def get_data(csv_sources, parent_dir):
    txl = []
    tyl = []
    tzl = []
    obsl = []
    obsw = []
    obsh = []

    pickle_dir_and_prefix = []

    def process(dirset):

        # load timestamps
        lidar_timestamps = dirset.dir + "/lidar_timestamps.csv"

        with open(lidar_timestamps) as csvfile:
            readCSV = csv.DictReader(csvfile, delimiter=',')

            for row in readCSV:

                ts = row['timestamp']

                pickle_dir_and_prefix.append(file_prefix_for_timestamp(dirset.dir, ts))
                txl.append(1.0)
                tyl.append(1.0)
                tzl.append(1.0)
                obsl.append(1.0)
                obsw.append(1.0)
                obsh.append(1.0)

    foreach_dirset(csv_sources, parent_dir, process)

    obs_centroid = [txl, tyl, tzl]
    obs_size = [obsl, obsw, obsh]
    return obs_centroid, pickle_dir_and_prefix, obs_size

#
# read input csv file to get the list of directories
#
def get_data_and_ground_truth(csv_sources, parent_dir):

    txl = []
    tyl = []
    tzl = []
    obsl = []
    obsw = []
    obsh = []

    pickle_dir_and_prefix = []

    def process(dirset):

        interp_lidar_fname = dirset.dir+"/obs_poses_interp_transform.csv"

        with open(interp_lidar_fname) as csvfile_2:
            readCSV_2 = csv.DictReader(csvfile_2, delimiter=',')

            for row2 in readCSV_2:
                ts = row2['timestamp']
                tx = row2['tx']
                ty = row2['ty']
                tz = row2['tz']

                pickle_dir_prefix = file_prefix_for_timestamp(dirset.dir, ts)
                pickle_dir_and_prefix.append(pickle_dir_prefix)
                txl.append(float(tx))
                tyl.append(float(ty))
                tzl.append(float(tz))
                obsl.append(float(dirset.mdr['l']))
                obsw.append(float(dirset.mdr['w']))
                obsh.append(float(dirset.mdr['h']))

    foreach_dirset(csv_sources, parent_dir, process)

    obs_centroid = [txl, tyl, tzl]
    obs_size = [obsl, obsw, obsh]
    return obs_centroid, pickle_dir_and_prefix, obs_size


def file_prefix_for_timestamp(parent_dir, ts=None):
    return parent_dir + "/lidar_360/" + (ts if ts is not None else '')

# ***** main loop *****
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load training data and ground truths")
    parser.add_argument("input_csv_file", type=str, default="data_folders.csv", help="data folder .csv")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")

    args = parser.parse_args()
    input_csv_file = args.input_csv_file
    dir_prefix = args.dir_prefix

    try:
        f = open(input_csv_file)
        f.close()
    except:
        print('Unable to read file: %s' % input_csv_file)
        f.close()
        sys.exit()

    # determine list of data sources and ground truths to load
    obs_centroids, pickle_dir_and_prefix, obs_size = get_data_and_ground_truth(input_csv_file, dir_prefix)

    # generate data in batches
    generator = data_generator_train(obs_centroids, obs_size, pickle_dir_and_prefix, 
        globals.BATCH_SIZE, globals.IMG_HEIGHT, globals.IMG_WIDTH, globals.NUM_CHANNELS, 
        globals.NUM_CLASSES, randomize=True)   
    
    images, obj_labels = next(generator)
    
    #print car pixels
    print("car pixels: ", len(np.where(obj_labels[:, :, 1] == 1)[1]))
    print("non-car pixels: ", len(np.where(obj_labels[:, :, 1] == 0)[1]))

