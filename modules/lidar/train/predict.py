import sys
import os
import argparse
import json
import datetime
import numpy as np
import cv2
import math
import csv
import rosbag
import sensor_msgs.point_cloud2

print(sys.path)
sys.path.append('../')
from process.globals import X_MIN, Y_MIN, RES, RES_RAD, LIDAR_MIN_HEIGHT
from scipy.ndimage.measurements import label
from globals import IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES, INPUT_SHAPE, BATCH_SIZE, PREDICTION_FILE_NAME
from loader import get_data, data_number_of_batches_per_epoch, data_generator_train, data_generator_predict
from model import build_model
from process.extract_rosbag_lidar import generate_lidar_2d_front_view

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
    #print(labels)

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

#
# take in find_obstacle() batch output (centroid x/theata,y/phi) and batch input(distance map) 
# and back project 2D centroid to 3D centroid(x,y,z)
#
def back_project_2D_2_3D(centroids, bboxes, distance_data, height_data):
    
    xyz_coor = np.zeros((centroids.shape[0],3))
    
    w = distance_data[0,:,:].shape[1]
    h = distance_data[0,:,:].shape[0]    
    
    valid_points_mask = np.logical_and(distance_data > 0, height_data > LIDAR_MIN_HEIGHT)
    
    ind_y, ind_x = np.unravel_index(np.arange(w*h),(h,w))
    ind_y_2d = np.reshape(ind_y,(h,w))
    ind_x_2d = np.reshape(ind_x,(h,w))        
        
    for i in range(centroids.shape[0]):
                
        if (not(valid_points_mask[i,int(centroids[i,1]), int(centroids[i,0])])):
        
            bb_left = int(bboxes[i,0])
            bb_right = int(bboxes[i,2])
            bb_top = int(bboxes[i,1])
            bb_bottom = int(bboxes[i,3])                             
                                    
            dist_x = ind_x_2d[bb_top:bb_bottom,bb_left:bb_right] - int(centroids[i,0])
            dist_y = ind_y_2d[bb_top:bb_bottom,bb_left:bb_right] - int(centroids[i,1])
            dist_2_centroid = np.sqrt(dist_x*dist_x + dist_y*dist_y)
            
            dist_2_centroid_valid = np.where(valid_points_mask[i, bb_top:bb_bottom,bb_left:bb_right], dist_2_centroid, 10e7)
            min_ind = np.argmin(dist_2_centroid_valid)
            min_val = np.min(dist_2_centroid_valid)   
            
            #print('min index {} min value {}'.format(min_ind, min_val))
            
            # cannot find any valid point.. zero out centroid and bounding box
            if (min_val == 10e7):
                centroids[i,1] = 0
                centroids[i,0] = 0  
                bboxes[i,0] = 0
                bboxes[i,1] = 0
                bboxes[i,2] = 0
                bboxes[i,3] = 0
                print('cannot find valid centroid')
            else:
                new_ind_1, new_ind_0 = np.unravel_index([min_ind],(bb_bottom-bb_top,bb_right-bb_left))
                new_ind_1 += bb_top
                new_ind_0 += bb_left
                print(' new centroid selected old: {} {} new: {} {}'.format(centroids[i,1], centroids[i,0], new_ind_1, new_ind_0))
                centroids[i,1], centroids[i,0] = new_ind_1, new_ind_0
                          
        distance = distance_data[i, int(centroids[i,1]), int(centroids[i,0])]
        height = height_data[i, int(centroids[i,1]), int(centroids[i,0])]
        theata = (centroids[i,0] + X_MIN) * RES_RAD[1]
              
        # increase  to approximate centroid - not surface of car
        distance += 0.75
        
        xyz_coor[i,0] = distance * math.cos(theata)
        xyz_coor[i,1] = - distance * math.sin(theata)
        xyz_coor[i,2] = height

        print('centroid: {}, height: {}, theta: {}, x: {}, y: {}, z: {}'.
             format(centroids[i], height, theata,
                    xyz_coor[i,0], xyz_coor[i,1], xyz_coor[i,2]))

    return xyz_coor


def write_prediction_data_to_csv(centroids, timestamps, output_file):

    csv_file = open(output_file, 'w')
    writer = csv.DictWriter(csv_file, ['timestamp', 'tx', 'ty', 'tz'])

    writer.writeheader()

    for centroid, ts in zip(centroids, timestamps):
        writer.writerow({'timestamp': ts, 'tx': centroid[0], 'ty': centroid[1], 'tz': centroid[2]})

def load_model(weightsFile):
    model = build_model(
        INPUT_SHAPE,
        NUM_CLASSES,
        trainable=False
    )

    print("reading existing weights")
    model.load_weights(weightsFile)
    
    return model
    
# return prection from a numpy array of point cloud        
def predict_point_cloud(model, points, cmap=None):
    points_2d = generate_lidar_2d_front_view(points, cmap=cmap)    
            
    input = np.ndarray(shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=float)
    input[:,:,0] = points_2d['distance_float']
    input[:,:,1] = points_2d['height_float']
    input[:,:,2] = points_2d['intensity_float']

    model_input = np.asarray([input])

    prediction = model.predict(model_input, verbose=1)
    centroid, _, _ = find_obstacle(prediction[0], INPUT_SHAPE)
    if centroid is None:
        centroid = (0, 0)
        
    centroids = np.array(centroid).reshape(1, 2)
    distance_data = np.array(points_2d['distance_float']).reshape(1, IMG_HEIGHT, IMG_WIDTH)
    height_data = np.array(points_2d['height_float']).reshape(1, IMG_HEIGHT, IMG_WIDTH)
    centroid_3d = back_project_2D_2_3D(centroids, distance_data, height_data)[0]
    #print('predicted centroid: {}'.format(centroid_3d))
    
    return centroid_3d

# return predictions from rosbag
def predict_rosbag(model, predict_file):
    bag = rosbag.Bag(predict_file, "r")        
    xyz_pred = []
    timestamps = []
    
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
        points = np.array(list(points))
                
        pred = predict_point_cloud(model, points)
        xyz_pred.append(pred)
        timestamps.append(t)
    
    return xyz_pred, timestamps       
  
# return predictions from lidar 2d frontviews  
def predict_lidar_frontview(model, predict_file, dir_prefix, export, output_dir):        
    # load data
    predict_data = get_data(predict_file, dir_prefix)

    n_batches_per_epoch = data_number_of_batches_per_epoch(predict_data[1], BATCH_SIZE)
    
    # get some data
    predictions = model.predict_generator(
        data_generator_train(
            predict_data[0], predict_data[2], predict_data[1],
            BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES,
            randomize=False
        ),
        n_batches_per_epoch,
        verbose=0
    )
    

    #print predict_data[0]
    # reload data as one big batch
    all_data_loader = data_generator_train(predict_data[0], predict_data[2], predict_data[1],
            len(predict_data[1]), IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS, NUM_CLASSES,
            randomize=False)
    all_images, all_labels = all_data_loader.next()
    

    bounding_boxes = np.zeros((all_images.shape[0],4))
    centroids = np.zeros((all_images.shape[0],2))
    timestamps = []
    print all_images.shape
    ind = 0


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

        timestamps.append(os.path.basename(file_prefix).split('_')[0])

        if centroid is not None:
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 2)            
            #print('{} -- centroid found: ({}, {}), area={}'.format(file_prefix, centroid[0], centroid[1], bbox_area))  
            centroids[ind][0] = centroid[0]
            centroids[ind][1] = centroid[1]
            bounding_boxes[ind,0] = bbox[0][0]
            bounding_boxes[ind,1] = bbox[0][1]
            bounding_boxes[ind,2] = bbox[1][0]
            bounding_boxes[ind,3] = bbox[1][1]               
        else:
            print('{} -- centroid not found'.format(file_prefix))
            centroids[ind][0] = 0
            centroids[ind][1] = 0
            bounding_boxes[ind,0] = 0
            bounding_boxes[ind,1] = 0
            bounding_boxes[ind,2] = 0
            bounding_boxes[ind,3] = 0               

        ind += 1

        if export:

            if output_dir is not None:
                file_prefix = output_dir + "/lidar_predictions/" + os.path.basename(file_prefix)
            else:
                file_prefix = os.path.dirname(file_prefix).replace('/lidar_360/', '') + "/lidar_predictions/" + os.path.basename(file_prefix)

            if not(os.path.isdir(os.path.dirname(file_prefix))):
                os.mkdir(os.path.dirname(file_prefix))

            cv2.imwrite(file_prefix + "_class.png", image)
        
    xyz_pred = back_project_2D_2_3D(centroids, bounding_boxes, all_images[:,:,:,0], all_images[:,:,:,1])
    
    return xyz_pred, timestamps
        
def main():
    parser = argparse.ArgumentParser(description='Lidar car/pedestrian trainer')
    parser.add_argument('weightsFile', type=str, default="", help='Model Filename')
    parser.add_argument("predict_file", type=str, default="", help="list of data folders for prediction or rosbag file name")
    parser.add_argument('--data_type', type=str, default="frontview", help='Data source for prediction: frontview, or rosbag')    
    parser.add_argument('--export', dest='export', action='store_true', help='Export images')
    parser.add_argument("--dir_prefix", type=str, default="", help="absolute path to folders")
    parser.add_argument('--output_dir', type=str, default=None, help='output file for prediction results')
    parser.set_defaults(export=False)

    args = parser.parse_args()
    output_dir = args.output_dir
    data_type = args.data_type
    predict_file = args.predict_file
    dir_prefix = args.dir_prefix
    
    if data_type is None or not data_type == 'rosbag':
        data_type = 'frontview'
    print('predicting from ' + data_type)    
    
    # load model with weights
    model = load_model(args.weightsFile)    
        
    if data_type == 'frontview':
        xyz_pred, timestamps = predict_lidar_frontview(model, predict_file, dir_prefix, args.export, output_dir)        
    else:
        xyz_pred, timestamps = predict_rosbag(model, predict_file)

    if output_dir is not None:
        file_prefix = output_dir + "/"

        write_prediction_data_to_csv(xyz_pred, timestamps, file_prefix + PREDICTION_FILE_NAME)
        print('prediction result written to ' + file_prefix + PREDICTION_FILE_NAME)    

        
if __name__ == '__main__':
    main()
