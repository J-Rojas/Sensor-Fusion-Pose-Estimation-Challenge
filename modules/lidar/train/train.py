import sys
import os
#sys.path.append('../../')
import argparse
import json
import datetime
import pandas as pd
import h5py

BATCH_SIZE = 32
IMG_WIDTH = 1801
IMG_HEIGHT = 32
NUM_CHANNELS = 2
NUM_LABELS = 2
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

import tensorflow as tf
from model import build_model
from loader import get_data_and_ground_truth, data_generator
from keras.callbacks import ModelCheckpoint, TensorBoard



def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument("--input_csv_file", type=str, default="../data/data_folders.csv", help="list of data folders for training")
    parser.add_argument('--modelFile', type=str, default="", help='Model Filename')
    parser.add_argument('--metadata', type=str, default="../data/metadata.csv", help='metadata filename')
    parser.add_argument('--outdir', type=str, default="./", help='output directory')
    
    args = parser.parse_args()
    input_csv_file = args.input_csv_file
    metadata_file = args.metadata
    outdir = args.outdir
    
    if (args.modelFile != "" ):
        with open(args.modelFile, 'r') as jfile:
            print("reading existing model and weights")
            model = model_from_json(json.loads(jfile.read()))
            weightsFile = args.modelFile.replace('json', 'h5')
            model.load_weights(weightsFile)
    else:
        model = build_model(INPUT_SHAPE)
        # save the model
        with open(os.path.join(outdir, 'lidar_model.json'), 'w') as outfile:
            json.dump(model.to_json(), outfile)
	
	metadata_df = pd.read_csv(metadata_file, header=0, index_col=None, quotechar="'")
    mdr = metadata_df.to_dict(orient='records')
    obs_size = [float(mdr[0]['l']), float(mdr[0]['w']), float(mdr[0]['h'])]
    
    # determine list of data sources and ground truths to load
    obs_centroid, pickle_dir_and_prefix = get_data_and_ground_truth(input_csv_file)
    
    print("start time:")
    print(datetime.datetime.now())
    
    checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'lidar_weights.{epoch:02d}-{loss:.4f}.hdf5'), verbose=1, save_weights_only=True)
    tensorboard = TensorBoard(histogram_freq=1, log_dir=os.path.join(outdir, 'tensorboard/'), write_graph=True, write_images=False)
    model.fit_generator(generator = data_generator(obs_centroid, obs_size, pickle_dir_and_prefix, 
        BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_LABELS), samples_per_epoch=1000, 
        nb_epoch=2, callbacks=[checkpointer,tensorboard])
    print("stop time:")
    print(datetime.datetime.now())
    
    # save model weights
    model.save_weights(os.path.join(outdir, "lidar_model.h5"), True)
    
if __name__ == '__main__':
    main()
