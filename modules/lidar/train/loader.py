import sys
sys.path.append('../')
import os
import numpy as np
import glob
import argparse
import csv
import sys
import random
import pickle
import pandas as pd
from common.csv_utils import foreach_dirset

from collections import defaultdict
from encoder import generate_label


def usage():
    print('Loads training data with ground truths and generate training batches')
    print('Usage: python loader.py --input_csv_file [csv file of data folders]')


def data_number_of_batches_per_epoch(data, BATCH_SIZE):
    size = len(data)
    return int(size / BATCH_SIZE) + (1 if size % BATCH_SIZE != 0 else 0)

#
# read in images/ground truths batch by batch
#
def data_generator_train(obs_centroids, obs_size, pickle_dir_and_prefix, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, randomize=True):
    tx = obs_centroids[0]
    ty = obs_centroids[1]
    tz = obs_centroids[2]
    obsl = obs_size[0]
    obsw = obs_size[1]
    obsh = obs_size[2]

    images = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    obj_labels = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT*IMG_WIDTH, NUM_CLASSES), dtype=np.uint8)

    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix, BATCH_SIZE)

    while 1:

        ziplist = list(zip(tx, ty, tz, obsl, obsw, obsh, pickle_dir_and_prefix))
        if randomize:
            random.shuffle(ziplist)
        tx, ty, tz, obsl, obsw, obsh, pickle_dir_and_prefix = zip(*ziplist)

        for batch in range(num_batches):

            load_lidar_data(images, pickle_dir_and_prefix, batch*BATCH_SIZE, BATCH_SIZE)
            load_label_data(obj_labels, tx, ty, tz, obsl, obsw, obsh,
                            (IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), batch*BATCH_SIZE, BATCH_SIZE)

            yield (images, obj_labels)


def data_generator_predict(pickle_dir_and_prefix, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES):

    images = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    num_batches = data_number_of_batches_per_epoch(pickle_dir_and_prefix, BATCH_SIZE)
    random.shuffle(pickle_dir_and_prefix)

    while 1:

        for ind in range(num_batches):
            load_lidar_data(images, pickle_dir_and_prefix, BATCH_SIZE)
            yield images


def load_lidar_data(images, pickle_dir_and_prefix, offset, size):

    batch_index = 0

    for ind in range(offset, offset + size):

        if ind < len(pickle_dir_and_prefix):

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


def load_label_data(obj_labels, tx, ty, tz, obsl, obsw, obsh, shape, offset, size):

    batch_index = 0

    for ind in range(offset, offset + size):

        if ind < len(tx):

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

    if len(sys.argv) < 2:
        usage()
        sys.exit()

    parser = argparse.ArgumentParser(description="Load training data and ground truths")
    parser.add_argument("--input_csv_file", type=str, default="data_folders.csv", help="data folder .csv")
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
    tx,ty,tz,pickle_dir_and_prefix = get_data_and_ground_truth(input_csv_file, dir_prefix)

    # generate data in batches
    generator = data_generator(tx, ty, tz, pickle_dir_and_prefix)
    image_distace, image_height, image_intensity, obj_location = next(generator)
    print(image_intensity)
