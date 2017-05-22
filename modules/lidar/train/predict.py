import sys
import os
import argparse
import json
import datetime
import numpy as np
import matplotlib.image as mpimg
from globals import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE
from loader import get_data_and_ground_truth, data_number_of_batches_per_epoch, data_generator_train
from model import build_model

from keras.models import model_from_json

def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument('weightsFile', type=str, default="", help='Model Filename')
    parser.add_argument("predict_file", type=str, default="", help="list of data folders for prediction")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--output_dir', type=str, default=None, help='output file for prediction image')

    args = parser.parse_args()
    output_dir = args.output_dir
    predict_file = args.predict_file
    dir_prefix = args.dir_prefix

    model = build_model(
        INPUT_SHAPE,
        NUM_CLASSES,
    )

    print("reading existing weights")
    model.load_weights(args.weightsFile)

    # load data
    predict_data = get_data_and_ground_truth(predict_file, dir_prefix)
    n_batches_per_epoch = data_number_of_batches_per_epoch(predict_data[1], BATCH_SIZE)

    predictions = model.predict_generator(
        data_generator_train(
            predict_data[0], predict_data[2], predict_data[1],
            BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES,
            randomize=False
        ),
        n_batches_per_epoch,
        verbose=0
    )

    # extract the 'car' category labels for all pixels in the first results, 0 is non-car, 1 is car
    for prediction, file_prefix in zip(predictions, predict_data[1]):

        classes = prediction[:, 1]

        classes = np.around(classes)

        print(np.where([classes == 1.0])[1])

        #print(classes, np.max(classes))

        obj_pixels = np.dstack((classes, classes, classes))
        obj_pixels = np.reshape(obj_pixels, INPUT_SHAPE)

        # generate output - white pixels for car pixels
        image = obj_pixels.astype(np.int)
        print(np.where([image[:, :, 0] == 1])[2])

        if output_dir is not None:
            file_prefix = output_dir + "/" + os.path.basename(file_prefix)
        else:
            file_prefix = os.path.dirname(file_prefix) + "/lidar_predictions/" + os.path.basename(file_prefix)

        mpimg.imsave(file_prefix + "_class.png", image)


if __name__ == '__main__':
    main()
