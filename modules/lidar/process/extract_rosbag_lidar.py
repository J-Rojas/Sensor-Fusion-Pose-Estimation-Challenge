#!/usr/bin/python
"""Extract lidar frontview images from a rosbag.
   usage: python extract_rosbag_lidar bag_file output_dir
"""

import os
import sys
import argparse
import numpy as np
import rosbag
import sensor_msgs.point_cloud2
import matplotlib.cm
import matplotlib.colors
import matplotlib.image as mpimg
import pickle

LIDAR_MAX_HEIGHT = 5
LIDAR_MIN_HEIGHT = -2
HRES = 0.35  # horizontal resolution (assuming 20Hz setting)
VRES = 0.4  # vertical res
VFOV = (-24.9, 2.0)  # Field of view (-ve, +ve) along vertical axis
Y_ADJUST = 25


def lidar_2d_front_view(points, res, fov, type, cmap = None, y_adjust=0.0):

    assert len(res) == 2, "res must be list/tuple of length 2"
    assert len(fov) == 2, "fov must be list/tuple of length 2"
    assert fov[0] <= 0, "first element in fov must be 0 or negative"
    assert type in ['intensity', 'height', 'distance'], "type must be 'intensity', 'height', or 'distance'"

    # gather the lidar data
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = points[:, 3]

    # L2 norm of X,Y dimension (distance from sensor)
    distance = np.sqrt(x ** 2 + y ** 2)
    l2_norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    v_fov_total = abs(fov[0] - fov[1])

    # radians
    res_rad = np.array(res) * (np.pi/180)

    # image coordinates
    x_img = np.arctan2(-y, x) / res_rad[1]
    y_img = np.arctan2(z, distance) / res_rad[0]

    # shift origin
    x_min = -360.0 / res[1] / 2
    x_img -= x_min
    x_max = 360.0 / res[1]

    y_min = fov[0] / res[0]
    y_img -= y_min
    y_max = v_fov_total / res[0]

    y_max += y_adjust

    colormap = matplotlib.cm.ScalarMappable(cmap=cmap) if cmap is not None else None
    min_val = 0

    if type == "intensity":
        pixel_values = r
    elif type == "height":
        pixel_values = z
        if cmap is not None:
            colormap = matplotlib.cm.ScalarMappable(cmap=cmap,
                                                    norm=matplotlib.colors.Normalize(
                                                        vmin=LIDAR_MIN_HEIGHT,
                                                        vmax=LIDAR_MAX_HEIGHT)
                                                    )
        min_val = LIDAR_MIN_HEIGHT
    elif type == 'distance':
        pixel_values = distance
    else:
        pixel_values = None

    y_img_int = y_img.astype(int)
    x_img_int = x_img.astype(int)
    img = np.ones((int(y_max) + 1, int(x_max) + 1)) * min_val
    norm = np.ones((int(y_max) + 1, int(x_max) + 1)) * 10000

    # should only keep point nearest to observer for duplicate x,y values
    for x, y, p, l in zip(x_img_int, y_img_int, pixel_values, l2_norm):
        y = min(y, int(y_max))
        y = max(y, 0)
        if norm[y, x] > l:
            img[y, x] = p
            norm[y, x] = l

    # flip pixels because y-axis increases as the laser angle increases downward
    img = np.flipud(img)

    retval = colormap.to_rgba(img, bytes=True, norm=True)[:,:,0:3] if cmap is not None else img

    return retval


def generate_lidar_2d_front_view(points, cmap=None):
    img_intensity, float_intensity = lidar_2d_front_view(points, res=(VRES, HRES), fov=VFOV, type='intensity', y_adjust=Y_ADJUST, cmap=cmap)
    img_distance, float_distance = lidar_2d_front_view(points, res=(VRES, HRES), fov=VFOV, type='distance', y_adjust=Y_ADJUST, cmap=cmap)
    img_height, float_height = lidar_2d_front_view(points, res=(VRES, HRES), fov=VFOV, type='height', y_adjust=Y_ADJUST, cmap=cmap)

    return {'intensity': img_intensity, 'distance': img_distance, 'height': img_height,
            'intensity_float': float_intensity, 'distance_float': float_distance, 'height_float': float_height}


def save_lidar_2d_images(output_dir, count, images):
    for k, img in images.iteritems():
        if k in ('intensity', 'distance', 'height'):
            mpimg.imsave('./{}/{}_{}.png'.format(output_dir, count, k), images[k], origin='upper')
        if k in ('intensity_float', 'distance_float', 'height_float'):
            f = open('./{}/{}_{}.lidar.p'.format(output_dir, count, k) , 'wb')
            pickle.dump(images[k], f)
            f.close()

def main():
    """Extract velodyne points and project to 2D images from a ROS bag
    """
    parser = argparse.ArgumentParser(description="Extract velodyne points and project to 2D images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--cmap", help="Color Map.", default='jet')

    args = parser.parse_args()
    bag_file = args.bag_file
    output_dir = args.output_dir 
    if not os.path.isfile(bag_file):
        print('bag_file ' + bag_file + ' does not exist')
        sys.exit()
        
    if not os.path.isdir(output_dir):
        print('output_dir ' + output_dir + ' does not exist')
        sys.exit()
        
    print("Extract velodyne_points from {} into {}".format(args.bag_file, args.output_dir))
    
    bag = rosbag.Bag(bag_file, "r")
    result = {'intensity': {}, 'distance': {}, 'height': {}}
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
        points = np.array(list(points))
        images = generate_lidar_2d_front_view(points, cmap=args.cmap)
        result['intensity'][str(t)] = images['intensity']
        result['distance'][str(t)] = images['distance']
        result['height'][str(t)] = images['height']
        #save_lidar_2d_images(output_dir, t, images)
        break
    bag.close()
    f = open(output_dir + '/lidar.p', 'wb')
    pickle.dump(result, f)
    f.close()
    
    
    
    #Load pickle: 
    #input  
    '''
    f = open(output_dir + '/lidar.p', 'rb')
    pickle_data = pickle.load(f)
    print(pickle_data)
    for mapType, pointMap in pickle_data.items():
        print(mapType)
        for t, value in pointMap.items():
            print(t)
            print(np.shape(value))
    '''
    
    #output
    '''
    distance
    1490149174663355139
    (93, 1029)
    ...
    intensity
    1490149174663355139
    (93, 1029)
    ...
    height
    1490149174663355139
    (93, 1029)
    ...
    
    '''
    return

if __name__ == '__main__':
    main()
