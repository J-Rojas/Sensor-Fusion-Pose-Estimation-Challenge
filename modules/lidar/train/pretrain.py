import sys
sys.path.append('../')
import cv2
import numpy as np
from encoder import generate_label
from common.csv_utils import foreach_dirset, load_lidar_interp

def calculate_sample_statistics(input_shape, lidar_gt, metadata):

    # generate label
    tx, ty, tz = np.float64(lidar_gt['tx']), np.float64(lidar_gt['ty']), np.float64(lidar_gt['tz'])
    rx, ry, rz = np.float64(lidar_gt['rx']), np.float64(lidar_gt['ry']), np.float64(lidar_gt['rz'])
    l, w, h = np.float64(metadata['l']), np.float64(metadata['w']), np.float64(metadata['h'])
    label = generate_label(tx, ty, tz, rx, ry, rz, l, w, h, input_shape)[:,1].flatten()

    # determine positive samples
    positive = len(np.where(label == 1)[0])

    # determine # of samples
    total_samples = input_shape[0] * input_shape[1]

    #print("positive = ", positive, ", total = ", total_samples)

    # assuming only one object per lidar frame, area == positive
    return {'positive': positive, 'total': total_samples, 'area': positive, 'count': 1}

def calculate_population_weights(input_csv_file, dir_prefix, input_shape):

    # cannot update variables with pass-by-value within closure, so use a dict to hold values by reference
    totals = {'positive': 0, 'samples': 0, 'area': 0, 'count': 0}

    def process(dirset):
        #load the ground truth for all lidar frames
        lidar_coord = load_lidar_interp(dirset.dir)

        for lidar_gt in lidar_coord:
            stats = calculate_sample_statistics(input_shape, lidar_gt, dirset.mdr)
            totals['positive'] += stats['positive']
            totals['samples'] += stats['total']
            totals['area'] += stats['area']
            totals['count'] += stats['count']

        print(totals)

    foreach_dirset(input_csv_file, dir_prefix, process)

    pton_ratio = np.float64(totals['positive']) / (totals['samples'] - totals['positive'])
    average_area = np.float64(totals['area']) / totals['count']

    return {'positive_to_negative_ratio': pton_ratio, 'average_area': average_area }
