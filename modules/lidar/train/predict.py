import sys
import os
import argparse
import json
import datetime
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from globals import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE
from loader import get_data_and_ground_truth, data_number_of_batches_per_epoch, data_generator_train
from model import build_model

from keras.models import model_from_json

MIN_PROB = 0.5 #lowest probability to participate in heatmap 
MIN_BBOX_AREA = 100 #minimum area of the bounding box to be declared a car
MIN_HEAT = 2 #lowest number of heat allowed

def find_obstacle(y_pred, input_shape):
    y_label = y_pred[:,1].flatten()      
    pred = np.reshape(y_label, input_shape[:2])
    ones = np.where(pred >= MIN_PROB)
    
    # Generate bounding box of size 4 for each positive pixel
    bbox_list = []
    for i in range(len(ones[0])):
        y = ones[0][i]
        x = ones[1][i]
        bbox = ((y - 2, x - 2), (y + 2, x + 2))
        bbox_list.append(bbox)
     
    heatmap = np.zeros_like(pred).astype(np.float)
    # Add heat to each box in box list
    for box in bbox_list:
        heatmap[box[0][0]:box[1][0], box[0][1]:box[1][1]] += 1
            
    # Apply threshold to help remove false positives
    heatmap[heatmap <= MIN_HEAT] = 0
    labels = label(heatmap)    

    # Iterate through all detected clusters
    max_area = 0
    for cluster_number in range(1, labels[1]+1):
        # Find pixels with each label value
        nonzero = (labels[0] == cluster_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #print(bbox)
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzeroy) - np.min(nonzeroy)
        area = width * height
        if area > max_area:
            largest_bbox = bbox
            max_area = area
    
    #print('max_area={}'.format(max_area))
    if max_area <= MIN_BBOX_AREA:
        return None, None, None
    
    largest_bbox = ((largest_bbox[0][0] + 2, largest_bbox[0][1] + 2), (largest_bbox[1][0] - 2, largest_bbox[1][1] - 2))
    centroid_x = int((largest_bbox[0][0] + largest_bbox[1][0]) / 2.0)
    centroid_y = int((largest_bbox[0][1] + largest_bbox[1][1]) / 2.0)
    
    return (centroid_x, centroid_y), largest_bbox, max_area
    
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

        #print(np.where([classes == 1.0])[1])
        #print(classes, np.max(classes))

        obj_pixels = np.dstack((classes, classes, classes))
        obj_pixels = np.reshape(obj_pixels, INPUT_SHAPE)

        # generate output - white pixels for car pixels
        image = obj_pixels.astype(np.uint8) * 255
                             
        centroid, bbox, bbox_area = find_obstacle(prediction, INPUT_SHAPE)       

        if centroid is not None:
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 2)            
            #print('{} -- centroid found: ({}, {}), area={}'.format(file_prefix, centroid[0], centroid[1], bbox_area))          
        else:
            print('{} -- centroid not found'.format(file_prefix))
            
        if output_dir is not None:
            file_prefix = output_dir + "/" + os.path.basename(file_prefix)
        else:
            file_prefix = os.path.dirname(file_prefix) + "/lidar_predictions/" + os.path.basename(file_prefix)

        cv2.imwrite(file_prefix + "_class.png", image)
        
        
if __name__ == '__main__':
    main()
