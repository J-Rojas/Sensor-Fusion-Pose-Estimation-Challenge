
# coding: utf-8
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import argparse
import math
import json
from process.extract_rosbag_lidar import X_MIN, Y_MIN, Y_MAX, RES_RAD, X_MAX, Y_ADJUST
from keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon, Circle

INPUT_SHAPE = (93, 1029, 2)

def project_2d(tx, ty, tz):
    d = np.sqrt(tx ** 2 + ty ** 2)
    #l2_norm = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    #print('d={} l2={}'.format(d, l2_norm))

    x_img = np.arctan2(-ty, tx) / RES_RAD[1]
    y_img = np.arctan2(tz, d) / RES_RAD[0]

    # shift origin
    x_img -= X_MIN
    y_img -= Y_MIN

    y_img = int(y_img)
    x_img = int(x_img)

    y_img = min(y_img, Y_MAX)
    y_img = max(y_img, 0)

    #return (y_img, x_img)
    return (x_img, y_img)

def get_bb(tx, ty, tz, l, w, h):
    bbox = []

    bbox.append(project_2d(tx-l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx-l/2., ty+w/2., tz-h/2.))
    bbox.append(project_2d(tx-l/2., ty-w/2., tz+h/2.))
    bbox.append(project_2d(tx-l/2., ty-w/2., tz-h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx+l/2., ty-w/2., tz+h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz-h/2.))
    bbox.append(project_2d(tx+l/2., ty-w/2., tz-h/2.))
    bbox = np.array(bbox)
    #print(bbox)
    return bbox

def generate_label(tx, ty, tz, l, w, h):
    bbox = get_bb(tx, ty, tz, l, w, h)

    upper_left_x = bbox.min(axis=0)[0]
    upper_left_y = bbox.min(axis=0)[1]
    lower_right_x = bbox.max(axis=0)[0]
    lower_right_y = bbox.max(axis=0)[1]
    print upper_left_x, upper_left_y, lower_right_x, lower_right_y

    label = np.ones(INPUT_SHAPE[:2])
    label[upper_left_x:lower_right_x, upper_left_y:lower_right_y] = 0
    y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle

    return y

def draw_bb(tx, ty, tz, l, w, h, infile, outfile):
    centroid = project_2d(tx, ty, tz)
    #print('Centroid: {}'.format(centroid))
    bbox = get_bb(tx, ty, tz, l, w, h)

    #infile = '../sample/10/out/10_distance.png'
    img = mpimg.imread(infile)
    fig, ax = plt.subplots(1, figsize=(20, 20))
    ax.imshow(img)
    ax.add_patch(Circle(centroid, 3, fill=True, color='red'))

    for p in bbox[:4]:
        ax.add_patch(Circle(p, 2, fill=True, color='white'))
    for p in bbox[-4:]:
        ax.add_patch(Circle(p, 2, fill=True, color='green'))

    ax.axis('off')
    #outfile = '../sample/10/out/10_distance_bb.png'
    fig.savefig(outfile)
    plt.close()

def test():
    #gps_l, gps_w, gps_h = (2.032, 0.7239, 1.6256)
    l, w, h = (4.2418,1.4478,1.5748)

    #centroid of obstacle after interplation
    #tx, ty, tz = (0.699597401296,-76.989,2.17780519741) #old 10.bag
    tx, ty, tz = (-0.8927325054898647, -3.6247593094278256, -0.648832347271497) #10.bag
    #tx, ty, tz = (-6.81401019142,-84.618,2.0329898085) #old 4_f.bag
    #tx, ty, tz = (9.083115901203417, 0.04826503520317882, -0.47151975040470145) #4_f.bag
    draw_bb(tx, ty, tz, l, w, h)

def main():
    parser = argparse.ArgumentParser(description="Draw bounding box (points) on projected 2D lidar images.")
    parser.add_argument("input_dir", help="Input directory.")
    parser.add_argument("output_dir", help="Output directory.")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        print('input_dir or output_dir does not exist')
        sys.exit()

    #input_dir needs to contain the following:
    #lidar_interpolated.csv, objects_obs1_rear_rtk.csv, and a sub directory lidar_360 that contains lidar images
    files = os.listdir(input_dir)
    for f in files:
        if f == 'lidar_df.csv':
            print('found ' + f)
            lidar_df = pd.read_csv(os.path.join(input_dir, 'lidar_df.csv'), index_col=None).reset_index()
        if f == 'obs_poses_interp_transformed.txt':
            print('found ' + f)
            obs_file = os.path.join(input_dir, 'obs_poses_interp_transformed.txt')
            #write to a json file to be read into a dataframe (needs double quotes around keys)
            obs_json_file = os.path.join(input_dir, 'obs_poses_interp_transformed_json.txt')

            with open(obs_file, 'r') as file:
                lines = '\n'.join(line.strip().replace('\'', '"') for line in file).strip()
                with open(obs_json_file, 'w') as out:
                    out.write(lines)

                obs1_df = pd.read_json(obs_json_file, lines=True).reset_index()

    if lidar_df is None or obs1_df is None:
        print('missing lidar_df.csv or obs_poses_interp_transformed.txt')
        sys.exit()

    obs1_df = obs1_df.join(lidar_df, on='index', rsuffix='_lidar')
    obs1_df.set_index(['timestamp'], inplace=True)
    all_ts = list(obs1_df.index)

    lidar_img_dir = os.listdir(os.path.join(input_dir, 'lidar_360'))
    l, w, h = (4.2418, 1.4478, 1.5748)

    for f in lidar_img_dir:
        if f.endswith('_distance.png'):
            ts = int(f.split('_')[0])

            if ts in list(obs1_df.index):
                tx = obs1_df.loc[ts]['tx']
                ty = obs1_df.loc[ts]['ty']
                tz = obs1_df.loc[ts]['tz']
                infile = os.path.join(input_dir, 'lidar_360', f)
                outfile = os.path.join(output_dir, f.split(".")[0] + '_bb.png')
                draw_bb(tx, ty, tz, l, w, h, infile, outfile)
            else:
                print('timestamp not in lidar_df {} not in index'.format(ts))


if __name__ == '__main__':
    main()
