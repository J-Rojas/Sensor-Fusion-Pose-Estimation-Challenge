import sys
sys.path.append('../')
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


def data_number_of_batches_per_epoch(obs_centroids, BATCH_SIZE):
    tx = obs_centroids[0]
    size = len(tx)
    return int(size / BATCH_SIZE)

#
# read in images/ground truths batch by batch
#
def data_generator(obs_centroids, obs_size, pickle_dir_and_prefix, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES):
    tx = obs_centroids[0]
    ty = obs_centroids[1]
    tz = obs_centroids[2]
    obsl = obs_size[0]
    obsw = obs_size[1]
    obsh = obs_size[2]

    images = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    obj_labels = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT*IMG_WIDTH, NUM_CLASSES), dtype=np.uint8)

    batch_index = 0
    num_batches = data_number_of_batches_per_epoch(obs_centroids, BATCH_SIZE)
    size = num_batches*BATCH_SIZE

    while 1:

        ziplist = list(zip(tx, ty, tz, obsl, obsw, obsh, pickle_dir_and_prefix))
        random.shuffle(ziplist)
        tx, ty, tz, obsl, obsw, obsh, pickle_dir_and_prefix = zip(*ziplist)

        for ind in range(size):

            fname = pickle_dir_and_prefix[ind]+"_distance_float.lidar.p"
            f = open(fname, 'rb')
            pickle_data = pickle.load(f)
            img_arr = np.asarray(pickle_data, dtype='float32')
            np.copyto(images[batch_index,:,:,0],img_arr)
            f.close();

            fname = pickle_dir_and_prefix[ind]+"_height_float.lidar.p"
            f = open(fname, 'rb')
            pickle_data = pickle.load(f)
            img_arr = np.asarray(pickle_data, dtype='float32')
            np.copyto(images[batch_index,:,:,1],img_arr)
            f.close();

            fname = pickle_dir_and_prefix[ind]+"_intensity_float.lidar.p"
            f = open(fname, 'rb')
            pickle_data = pickle.load(f)
            img_arr = np.asarray(pickle_data, dtype='float32')
            np.copyto(images[batch_index,:,:,2],img_arr)
            f.close()

            label = generate_label(tx[ind], ty[ind], tz[ind], obsl[ind], obsw[ind], obsh[ind],(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES))
            #label = np.ones(shape=(IMG_HEIGHT, IMG_WIDTH),dtype=np.dtype('u2'))
            np.copyto(obj_labels[batch_index], np.uint8(label))

            batch_index = batch_index + 1

            if (batch_index >= BATCH_SIZE):
                batch_index = 0
                yield (images, obj_labels)


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

                pickle_dir_prefix = dirset.dir+"/lidar_360/"+ts
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
