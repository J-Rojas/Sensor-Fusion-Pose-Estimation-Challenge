import sys
import os
import argparse
import json
import datetime
import numpy as np
import pandas as pd
import h5py
from globals import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, \
                    NUM_CHANNELS, NUM_CLASSES, EPOCHS, INPUT_SHAPE, K_NEGATIVE_SAMPLE_RATIO_WEIGHT
from pretrain import calculate_population_weights

import tensorflow as tf
from model import build_model, load_model
from loader import get_data_and_ground_truth, data_generator_train, data_number_of_batches_per_epoch
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
import keras.backend as K
import matplotlib.pyplot as plt



def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    #y_pred = tf.Print(y_pred, ["preds", y_pred])
    #y_true = tf.Print(y_true, ["labels", y_true])

    labels_bkg, labels_frg = tf.split(y_true, 2, 2)
    preds_bkg, preds_frg = tf.split(y_pred, 2, 2)

    true_positives = K.sum(K.round(K.clip(labels_frg * preds_frg, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(preds_frg, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """

    labels_bkg, labels_frg = tf.split(y_true, 2, 2)
    preds_bkg, preds_frg = tf.split(y_pred, 2, 2)

    true_positives = K.sum(K.round(K.clip(labels_frg * preds_frg, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(labels_frg, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.precisions = []
        self.recalls = []
        self.val_losses = []
        self.val_precisions = []
        self.val_recalls = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.precisions.append(logs.get('precision'))
        self.recalls.append(logs.get('recall'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_precisions.append(logs.get('val_precision'))
        self.val_recalls.append(logs.get('val_recall'))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.precisions.append(logs.get('precision'))
        self.recalls.append(logs.get('recall'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_precisions.append(logs.get('val_precision'))
        self.val_recalls.append(logs.get('val_recall'))


def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument("--train_file", type=str, default="../data/train_folders.csv",
                        help="list of data folders for training")
    parser.add_argument("--val_file", type=str, default="../data/validation_folders.csv",
                        help="list of data folders for validation")
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--modelFile', type=str, default="", help='Model Filename')
    parser.add_argument('--weightsFile', type=str, default="", help='Weights Filename')
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

    metrics = [recall, precision]

    if args.modelFile != "":
        weightsFile = args.modelFile.replace('json', 'h5')
        if args.weightsFile != "":
            weightsFile = args.weightsFile
        model = load_model(args.modelFile, weightsFile,
                           INPUT_SHAPE, NUM_CLASSES,
                           obj_to_bkg_ratio=population_statistics_train[
                                                'positive_to_negative_ratio'] * K_NEGATIVE_SAMPLE_RATIO_WEIGHT,
                           avg_obj_size=population_statistics_train['average_area'],
                           metrics=metrics
                           )
    else:
        model = build_model(
            INPUT_SHAPE,
            NUM_CLASSES,
            obj_to_bkg_ratio=population_statistics_train['positive_to_negative_ratio'] * K_NEGATIVE_SAMPLE_RATIO_WEIGHT,
            avg_obj_size=population_statistics_train['average_area'],
            metrics=metrics
        )
        # save the model
        with open(os.path.join(outdir, 'lidar_model.json'), 'w') as outfile:
            json.dump(model.to_json(), outfile)

    # determine list of data sources and ground truths to load
    train_data = get_data_and_ground_truth(train_file, dir_prefix)
    val_data = get_data_and_ground_truth(validation_file, dir_prefix)

    # number of batches per epoch
    n_batches_per_epoch_train = data_number_of_batches_per_epoch(train_data[1], BATCH_SIZE)
    n_batches_per_epoch_val = data_number_of_batches_per_epoch(val_data[1], BATCH_SIZE)

    print("Number of batches per epoch: {}".format(n_batches_per_epoch_train))
    print("start time:")
    print(datetime.datetime.now())

    checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'lidar_weights.{epoch:02d}-{loss:.4f}.hdf5'), verbose=1, save_weights_only=True)
    tensorboard = TensorBoard(histogram_freq=1, log_dir=os.path.join(outdir, 'tensorboard/'), write_graph=True, write_images=False)
    loss_history = LossHistory()
    try:
        model.fit_generator(
            data_generator_train(
                train_data[0], train_data[2], train_data[1],
                BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
            ),  # generator
            n_batches_per_epoch_train,  # number of batches per epoch
            validation_data=data_generator_train(
                val_data[0], val_data[2], val_data[1],
                BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES
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
    model.save_weights(os.path.join(outdir, "lidar_model.h5"), True)

    #print precision_recall_array
    plot_precision_recall_curve(loss_history.precisions,loss_history.recalls,outdir)


def plot_filtered_PR_curve(outdir,data_filename, maxgap=0.03):
    fname = '{0}/{1}'.format(outdir,data_filename)#precision_recall_value_100epoch
    df = pd.read_csv(fname,header=None,names=['precision','recall'])
    df['recall_grp'] = (df['recall']/maxgap).astype(int)*maxgap

    aggregation = {'recall':{
                             'mean_recall':'mean'
                             },
                   'precision':{ 'max_precision' : lambda x: max(x),
                                 'min_precision': lambda x: min(x),
                                 'mean_precision': 'mean'
                                 }}

    grouped_recall_with_precision = df.groupby('recall_grp').agg(aggregation)
    plt.clf()
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'max_precision')], lw=2, color='teal', label='Max Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'mean_precision')], lw=2, color='cornflowerblue',label='Mean Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'min_precision')], lw=2, color='turquoise', label='Min Precision')

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Filtered Precision-Recall Curve, filterGap={0}'.format(maxgap))
    plt.legend(loc="upper right")
    fname = '{0}/precision_recall_curve_filtered.png'.format(outdir)
    plt.savefig(fname)
    print 'Filtered Precision-Recall Curve saved at ' + fname

def plot_precision_recall_curve(precision_arr, recall_arr,outdir):

    print 'Ploting Precision-Recall Curve...'
    data = np.array(zip(precision_arr, recall_arr), dtype=[('precision', float), ('recall', float)])
    data_filename = 'precision_recall_value.csv'
    np.savetxt('{0}/{1}'.format(outdir,data_filename), data, delimiter=",")
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(data['recall'], data['precision'], lw=2, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")

    fname = '{0}/precision_recall_curve.png'.format(outdir)
    plt.savefig(fname)
    print 'Precision-Recall Curve saved at '+ fname
    #plt.show()

    plot_filtered_PR_curve(outdir, data_filename)


if __name__ == '__main__':
    main()
