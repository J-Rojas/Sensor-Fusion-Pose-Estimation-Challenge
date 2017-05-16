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
import cv2
from process.extract_rosbag_lidar import X_MIN, Y_MIN, Y_MAX, RES_RAD, X_MAX
from keras.utils import to_categorical


print(Y_MIN, Y_MAX, RES_RAD)

def project_2d(tx, ty, tz):
    d = np.sqrt(tx ** 2 + ty ** 2)
    l2_norm = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)

    x_img = np.arctan2(-ty, tx) / RES_RAD[1]
    y_img = np.arcsin(tz/l2_norm) / RES_RAD[0]

    #print('tx={}, ty={}, tz={}, d={} l2={} arcsin={}'.format(tx,ty,tz,d, l2_norm, np.arcsin(tz/l2_norm)))

    # shift origin
    x_img -= X_MIN
    y_img -= Y_MIN

    y_img = int(y_img)
    x_img = int(x_img)

    y_img = min(y_img, Y_MAX)
    y_img = max(y_img, 0)

    y_img = int(Y_MAX - y_img)

    #return (y_img, x_img)
    return (x_img, y_img)

#returns the projected corners in order of distance from centroid in 2d
def get_bb(tx, ty, tz, l, w, h):
    bbox = []

    bbox.append(project_2d(tx-l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx-l/2., ty+w/2., tz-h/2.))
    bbox.append(project_2d(tx-l/2., ty-w/2., tz+h/2.))
    bbox.append(project_2d(tx-l/2., ty-w/2., tz-h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz-h/2.))
    bbox.append(project_2d(tx+l/2., ty-w/2., tz+h/2.))
    bbox.append(project_2d(tx+l/2., ty-w/2., tz-h/2.))

    #print(bbox)
    bbox = np.array(bbox)
    centroid = project_2d(tx, ty, tz)
    d = []
    for p in bbox:
        d.append(distance(centroid, (p[0], p[1])))

    d = np.array(d)
    indices = np.argsort(d)

    sorted_corners = bbox[indices]
    return sorted_corners

#distance between two points in 2D
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_inner_rect(tx, ty, tz, l, w, h):
    bbox = get_bb(tx, ty, tz, l, w, h)
    sorted_corners = bbox[:4]

    upper_left_x = sorted_corners.min(axis=0)[0]
    upper_left_y = sorted_corners.min(axis=0)[1]
    lower_right_x = sorted_corners.max(axis=0)[0]
    lower_right_y = sorted_corners.max(axis=0)[1]
    return (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)


def get_outer_rect(tx, ty, tz, l, w, h):
    bbox = get_bb(tx, ty, tz, l, w, h)
    sorted_corners = bbox[-4:]

    upper_left_x = sorted_corners.min(axis=0)[0]
    upper_left_y = sorted_corners.min(axis=0)[1]
    lower_right_x = sorted_corners.max(axis=0)[0]
    lower_right_y = sorted_corners.max(axis=0)[1]
    return (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)


def generate_label_from_circle(tx, ty, tz, l, w, h, INPUT_SHAPE):
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, l, w, h)
    r = min((lower_right_y - upper_left_y) / 2.0, (lower_right_x - upper_left_x) / 2.0)
    centroid = project_2d(tx, ty, tz)

    label = np.ones(INPUT_SHAPE[:2])

    #print(upper_left_x, lower_right_x)
    #print(upper_left_y, lower_right_y)

    for x in range(upper_left_x, lower_right_x, 1):
        for y in range(upper_left_y, lower_right_y, 1):
            if distance(centroid, (x, y)) <= r:
                label[y, x] = 0

    # label[upper_left_x:lower_right_x, upper_left_y:lower_right_y] = 0
    y = to_categorical(label, num_classes=2)  # 1st dimension: on-vehicle, 2nd dimension: off-vehicle

    return y


def generate_label(tx, ty, tz, l, w, h, INPUT_SHAPE, method='circle'):
    if method == 'circle':
        y = generate_label_from_circle(tx, ty, tz, l, w, h, INPUT_SHAPE)
    else:
        if method == 'inner_rect':
            (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, l, w, h)
        elif method == 'outer_rect':
            (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_outer_rect(tx, ty, tz, l, w, h)
        #print (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)

        label = np.ones(INPUT_SHAPE[:2])
        label[upper_left_y:lower_right_y, upper_left_x:lower_right_x] = 0
        y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle

    #print(np.nonzero(y[:,0])[0].shape[0]) #number of on-vehicle pixels
    return y


def draw_bb_circle(tx, ty, tz, l, w, h, infile, outfile):
    centroid = project_2d(tx, ty, tz)
    #print('Centroid: {}'.format(centroid))
    bbox = get_bb(tx, ty, tz, l, w, h)
    (upper_left_x, upper_left_y), (lower_right_x, lower_right_y) = get_inner_rect(tx, ty, tz, l, w, h)
    #print (upper_left_x, upper_left_y), (lower_right_x, lower_right_y)
    r = min((lower_right_y - upper_left_y)/2.0, (lower_right_x - upper_left_x)/2.0)
    #print r

    img = cv2.imread(infile)
    cv2.circle(img, centroid, 2, (0, 0, 255), thickness=-1)

    for p in bbox[:4]:
        cv2.circle(img, (p[0], p[1]), 2, (255, 255, 255), thickness=-1)
    for p in bbox[-4:]:
        cv2.circle(img, (p[0], p[1]), 2, (0, 255, 0), thickness=-1)

    cv2.circle(img, centroid, int(r), (0, 255, 255), thickness=2)

    #save image
    cv2.imwrite(outfile, img)


def draw_bb_rect(tx, ty, tz, l, w, h, infile, outfile, method='inner_rect'):
    centroid = project_2d(tx, ty, tz)
    img = cv2.imread(infile)
    cv2.circle(img, centroid, 2, (0, 0, 255), thickness=-1)

    bbox = get_bb(tx, ty, tz, l, w, h)
    for p in bbox:
        cv2.circle(img, (p[0], p[1]), 2, (255, 255, 255), thickness=-1)

    if method == 'inner_rect':
        upper_left, lower_right = get_inner_rect(tx, ty, tz, l, w, h)
    elif method == 'outer_rect':
        upper_left, lower_right = get_outer_rect(tx, ty, tz, l, w, h)

    cv2.rectangle(img, upper_left, lower_right, (0, 255, 0), 1)

    #save image
    cv2.imwrite(outfile, img)


def draw_bb(tx, ty, tz, l, w, h, infile, outfile, method='circle'):
    if method == 'circle':
        draw_bb_circle(tx, ty, tz, l, w, h, infile, outfile)
    else:
        draw_bb_rect(tx, ty, tz, l, w, h, infile, outfile, method)


def test():
    #gps_l, gps_w, gps_h = (2.032, 0.7239, 1.6256)
    l, w, h = (4.2418,1.4478,1.5748)

    #centroid of obstacle after interplation
    #tx, ty, tz = (0.699597401296,-76.989,2.17780519741) #old 10.bag
    tx, ty, tz = (-0.8927325054898647, -3.6247593094278256, -0.648832347271497) #10.bag
    #tx, ty, tz = (-6.81401019142,-84.618,2.0329898085) #old 4_f.bag
    #tx, ty, tz = (9.083115901203417, 0.04826503520317882, -0.47151975040470145) #4_f.bag
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_circle.png', 'circle')
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_inner.png', 'inner_rect')
    draw_bb(tx, ty, tz, l, w, h, '../sample/10/out/lidar_360/1490991699437114271_distance.png', '../sample/10_1490991699437114271_distance_outer.png', 'outer_rect')


def main():
    parser = argparse.ArgumentParser(description="Draw bounding box on projected 2D lidar images.")
    parser.add_argument("input_dir", help="Input directory.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("shape", help="bounding box shape: circle, outer_rect, inner_rect")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    shape = args.shape

    if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
        print('input_dir or output_dir does not exist')
        sys.exit()

    if shape not in ('circle', 'outer_rect', 'inner_rect'):
        print('shape must be one of the following: circle, outer_rect, inner_rect')
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

    print(obs1_df)

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
                draw_bb(tx, ty, tz, l, w, h, infile, outfile, method=shape)
            else:
                print('timestamp not in lidar_df {} not in index'.format(ts))


if __name__ == '__main__':
    main()
