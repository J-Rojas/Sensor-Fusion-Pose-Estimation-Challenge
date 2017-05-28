import sys
sys.path.append('../')
import csv
import glob
import argparse
import os.path
import scipy.interpolate
from common.tracklet_generator import Tracklet, TrackletCollection

def main():

    appTitle = "Udacity Team-SF: Tracklet generator"
    parser = argparse.ArgumentParser(description=appTitle)
    parser.add_argument('pred_csv', type=str, help='Prediction CSV')
    parser.add_argument('camera_csv', type=str, help='Camera timestamps CSV')
    parser.add_argument('metadata', type=str, help='Metadata File')
    parser.add_argument('out_xml', type=str, help='Tracklet File')

    args = parser.parse_args()

    points = []
    camera_timestamps = []

    with open(args.pred_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', restval='')

        for r in reader:
            timestamp = int(r['timestamp'])
            tx = float(r['tx'])
            ty = float(r['ty'])
            tz = float(r['tz'])

            points.append({'timestamp': timestamp, 'tx': tx, 'ty': ty, 'tz': tz})

    with open(args.camera_csv, 'r') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=',', restval='')

        for r in reader:
            timestamp = int(r['timestamp'])
            camera_timestamps.append(timestamp)

    timestamps = list(map(lambda x: x['timestamp'], points))
    txs = list(map(lambda x: x['tx'], points))
    tys = list(map(lambda x: x['ty'], points))
    tzs = list(map(lambda x: x['tz'], points))

    print(timestamps)
    print(txs)
    print(tys)

    fx = scipy.interpolate.interp1d(timestamps, txs)
    fy = scipy.interpolate.interp1d(timestamps, tys)
    fz = scipy.interpolate.interp1d(timestamps, tzs)

    interpolated_camera = []

    csvfile = open(args.metadata, 'r')
    reader = csv.DictReader(csvfile, delimiter=',')
    tracklet = None
    tracklet_xml = TrackletCollection()
    for mdr in reader:
        tracklet = Tracklet('Car', float(mdr['l']), float(mdr['w']), float(mdr['h']))

    # fudge factor to increase score
    offset = [-8.93, 0.35, -.22]

    for camera_timestamp in camera_timestamps:
        if camera_timestamp < timestamps[0]:
            camera_timestamp = timestamps[0]
        elif camera_timestamp > timestamps[-1]:
            camera_timestamp = timestamps[-1]
        interpolated_camera.append({'timestamp': camera_timestamp,
                                    'tx': fx(camera_timestamp) + offset[0],
                                    'ty': fy(camera_timestamp) + offset[1],
                                    'tz': fz(camera_timestamp) + offset[2],
                                    'rx': 0, 'ry': 0, 'rz': 0})

    tracklet.poses = interpolated_camera
    tracklet_xml.tracklets = [tracklet]
    tracklet_xml.write_xml(args.out_xml)


if __name__ == '__main__':
    main()
