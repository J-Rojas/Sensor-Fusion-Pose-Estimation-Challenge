import argparse
import sys
import numpy as np
import pygame
import rosbag
import datetime
import sensor_msgs.point_cloud2
import matplotlib as mpl
mpl.use('Agg')  #Skip using X11
import matplotlib.pyplot as plt
from itertools import repeat
import cv2


startsec = 0

# vertical slice numbers
m = 20

# unit is meter
max_range = 120
width_grid_length = 0.1
height_grid_length = 0.1


def generate_zeros(height, width, grid_length):
    s = (int(height/grid_length), int(width/grid_length))
    return np.zeros(s)


def get_z_range(points):
    result = [sys.float_info.max, sys.float_info.min]
    for point in points:
        p_z = point[2]
        result[0] = p_z if result[0] > p_z else result[0]
        result[1] = p_z if result[1] < p_z else result[1]
    
    #dirty patch to avoid out of index error for the top point, so it will fall into the slice[m - 1], instead of slice[m]
    result[1] = result[1] + 0.00001
    
    print("Min Height ", result[0], "Max Height ", result[1])
    return result


def get_cell_num(range, grid_length, m):
    return (int(range*2/grid_length)**2)* m


# generate index for z, x, y axis
def generate_index(point, x_range, y_range, z_range, m, width_grid_length, height_grid_length):
    return (int( (point[2] - z_range[0])/((z_range[1] - z_range[0])/m)), int((point[0] - x_range[0])/width_grid_length), int((point[1] - y_range[0])/height_grid_length))
   
 
def generate_point_channel(x_range, y_range, width_grid_length, height_grid_length):
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    return [ [ [] for i in repeat(None, int(width/width_grid_length)) ] for i in repeat(None, int(height/height_grid_length))]


def generate_value_channel(x_range, y_range, width_grid_length, height_grid_length):
    width = x_range[1] - x_range[0]
    height = y_range[1] - y_range[0]
    return [ [ 0 for i in repeat(None, int(width/width_grid_length)) ] for i in repeat(None, int(height/height_grid_length))]


def produce_view(channel, name):
    fig = plt.figure()
    plt.imshow(channel)
    fig.savefig(name + '_plot.png')


# The point cloud density
# indicates the number of points in each cell. To normalize
# the feature, it is computed as min(1.0, log(N+1) log(64) ), where N is the number of points in the cell.
def normalize(channel):
    for i, r in enumerate(channel):
        for j, c in enumerate(r):
            channel[i][j] = 255 * min(np.log(channel[i][j] + 1) * 8, 1)
    return channel
    

# generate birds view for one frame
def generate_birds_eye_view(points):
    # min and max of width and heigth
    x_range = (-max_range, max_range)
    y_range = (-max_range, max_range)

    #count in the cell
    density_channel = generate_value_channel(x_range, y_range, width_grid_length, height_grid_length)
    print(np.shape(density_channel))
    #max reflective value in the cell
    intensity_channel = generate_value_channel(x_range, y_range, width_grid_length, height_grid_length)

    #m sliced channels for computing the height maps
    height_map_channel = [ generate_value_channel(x_range, y_range, width_grid_length, height_grid_length) for i in repeat(None, m) ]
    print(np.shape(height_map_channel))
    z_range = get_z_range(points)
    for point in points:
        index = generate_index(point, x_range, y_range, z_range, m, width_grid_length, height_grid_length) # generate index for z, x, y axis
        
        #print(index)
        density_channel[index[1]][index[2]] = density_channel[index[1]][index[2]] + 1 #density_channel is count in the cell
        
        # intensity_channel is max reflective value in the cell
        intensity_channel[index[1]][index[2]] = intensity_channel[index[1]][index[2]] if intensity_channel[index[1]][index[2]] > point[3] else point[3]
        
        # height_map_channel is the highest z value in the cell in its own slice
        height_map_channel[index[0]][index[1]][index[2]] = height_map_channel[index[0]][index[1]][index[2]] if height_map_channel[index[0]][index[1]][index[2]] > point[2] else point[2]
    produce_view(intensity_channel, "intensity_channel")
    produce_view(normalize(density_channel), "density_channel")
    for i, c in enumerate(height_map_channel):
        produce_view(height_map_channel[i], "height_map_channel_" + str(i))

def load(topic, msg, time):
    t = time.to_sec()
    since_start = msg.header.stamp.to_sec()-startsec
    arrPoints = []
    if topic in ['/radar/points','/velodyne_points']:
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=True)
        #print(len(points))
        for point in points:
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]
            arrPoints.append(point[:4])
        #print(arrPoints)
    return arrPoints

 
def read(dataset, skip, topics_list):
    """
    return an image of 
    """
    startsec = 0

    print("reading rosbag ", dataset)
    bag = rosbag.Bag(dataset, 'r')
    for topic, msg, t in bag.read_messages(topics=topics_list):
      if startsec == 0:
          startsec = t.to_sec()
          if skip < 24*60*60:
              skipping = t.to_sec() + skip
              print("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
          else:
              skipping = skip
              print("skipping to ", skip, " from ", startsec, " ...")
      else:
          if t.to_sec() > skipping:
              points = load(topic, msg, t)
              generate_birds_eye_view(points)
              break;    #firstly try one frame

# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="appTitle")
    parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
    parser.add_argument('--skip', type=int, default="0", help='skip seconds')
    args = parser.parse_args()
    dataset = args.dataset
    skip = args.skip

    topics_list = [
      #'/image_raw',
      #'/gps/fix',
      #'/radar/points',
      '/velodyne_points'
      #'/radar/range'
    ]
    read(dataset, skip, topics_list)
