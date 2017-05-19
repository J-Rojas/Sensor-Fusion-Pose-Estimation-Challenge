import sys
import os
import argparse
import json
import datetime
import pandas as pd
import h5py
from pretrain import calculate_population_weights

BATCH_SIZE = 32
EPOCHS = 100
IMG_WIDTH = 1801
IMG_HEIGHT = 32
NUM_CHANNELS = 3
NUM_CLASSES = 2
K_NEGATIVE_SAMPLE_RATIO_WEIGHT = 4
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

import tensorflow as tf
from model import build_model
from loader import get_data_and_ground_truth, data_generator, data_number_of_batches_per_epoch
from keras.callbacks import ModelCheckpoint, TensorBoard


def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument("--train_file", type=str, default="../data/train_folders.csv",
                        help="list of data folders for training")
    parser.add_argument("--val_file", type=str, default="../data/validation_folders.csv",
                        help="list of data folders for validation")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--modelFile', type=str, default="", help='Model Filename')
    parser.add_argument('--outdir', type=str, default="./", help='output directory')

    args = parser.parse_args()
    train_file = args.train_file
    validation_file = args.val_file
    outdir = args.outdir
    dir_prefix = args.dir_prefix

    # calculate population statistic - they are only calculated for the training set since the weights will remain
    # unchanged in the validation/test set
    population_statistics_train = calculate_population_weights(train_file, dir_prefix, INPUT_SHAPE)
    print("Train statistics: ", population_statistics_train)

    if args.modelFile != "":
        with open(args.modelFile, 'r') as jfile:
            print("reading existing model and weights")
            model = model_from_json(json.loads(jfile.read()))
            weightsFile = args.modelFile.replace('json', 'h5')
            model.load_weights(weightsFile)
    else:
        model = build_model(
            INPUT_SHAPE,
            NUM_CLASSES,
            obj_to_bkg_ratio=population_statistics_train['positive_to_negative_ratio'] * K_NEGATIVE_SAMPLE_RATIO_WEIGHT,
            avg_obj_size=population_statistics_train['average_area']
        )
        # save the model
        with open(os.path.join(outdir, 'lidar_model.json'), 'w') as outfile:
            json.dump(model.to_json(), outfile)

    # determine list of data sources and ground truths to load
    train_data = get_data_and_ground_truth(train_file, dir_prefix)
    val_data = get_data_and_ground_truth(validation_file, dir_prefix)

    # number of batches per epoch
    n_batches_per_epoch_train = data_number_of_batches_per_epoch(train_data[0], BATCH_SIZE)
    n_batches_per_epoch_val = data_number_of_batches_per_epoch(val_data[0], BATCH_SIZE)

    print("start time:")
    print(datetime.datetime.now())

    checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'lidar_weights.{epoch:02d}-{loss:.4f}.hdf5'), verbose=1, save_weights_only=True)
    tensorboard = TensorBoard(histogram_freq=1, log_dir=os.path.join(outdir, 'tensorboard/'), write_graph=True, write_images=False)
    model.fit_generator(
        data_generator(
            train_data[0], train_data[2], train_data[1],
            BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
        ),  # generator
        n_batches_per_epoch_train,  # number of batches per epoch
        validation_data=data_generator(
            val_data[0], val_data[2], val_data[1],
            BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
        ),
        validation_steps=n_batches_per_epoch_val,  # number of batches per epoch
        epochs=EPOCHS,
        callbacks=[checkpointer, tensorboard],
        verbose=1
    )
    print("stop time:")
    print(datetime.datetime.now())

    # save model weights
    model.save_weights(os.path.join(outdir, "lidar_model.h5"), True)

if __name__ == '__main__':
    main()
