import sys
sys.path.append('../../didi_competition/tracklets/python/')
import os
import json
import rosbag
import functools
import pandas as pd
import PyKDL as kd
import numpy as np
import argparse
from xml.etree.ElementTree import fromstring
from xmljson import Parker

from collections import defaultdict
from bag_to_kitti import estimate_obstacle_poses, rtk2dict, interpolate_to_camera
from tracket_parser import clean_items_list, put_timestamps_with_frame_ids

# notes
# convert stamps to .to_nsec() in extract_rosbag.py while saving lidar images
#

#
# lidar to dictionary
#
def lidar2dict(msg, stamp, lidar_dict):
    lidar_dict["timestamp"].append(stamp.to_nsec())
    lidar_dict["cap_lidar"].append("cap_lidar")
    #lidar_dict["timestamp_lidar"].append(stamp.to_nsec())


#
# coordinate transformation of obstacle centroid from base gps to lidar
#
def obstacle_coordinate_base2lidar(obs_rear_rtk, cap_rear_rtk, cap_front_rtk, mdr):

    lrg_to_gps = [mdr[0]['gps_l'], -mdr[0]['gps_w'], mdr[0]['gps_h']]
    lrg_to_centroid = [mdr[0]['l'] / 2., -mdr[0]['w'] / 2., mdr[0]['h'] / 2.]
    gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)

    obs_poses = estimate_obstacle_poses( cap_front_rtk, cap_rear_rtk, obs_rear_rtk, gps_to_centroid)

    return obs_poses


#
# fill in obstacle position in lidar data with obstacle rtk data
#
def interpolate_lidar_with_rtk(bag_filename, metadata_filename, outdir):

    # read tracklet file and collect timestamp to check for correct transformation
    cam_timestamps = []

    metadata_df = pd.read_csv(metadata_filename, header=0, index_col=None, quotechar="'")
    mdr = metadata_df.to_dict(orient='records')

    bag = rosbag.Bag(bag_filename, 'r')
    topicTypesMap = bag.get_type_and_topic_info().topics

    obs_rear_rtk_cols = ["obs_rear_rtk", "timestamp", "tx", "ty", "tz", "rx", "ry", "rz", "t_tx", "t_ty", "t_tz"]
    obs_rear_rtk_dict = defaultdict(list)

    cap_rear_rtk_cols = ["cap_rear_rtk", "timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
    cap_rear_rtk_dict = defaultdict(list)

    cap_front_rtk_cols = ["cap_front_rtk", "timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
    cap_front_rtk_dict = defaultdict(list)

    lidar_cols = ["cap_lidar", "timestamp"]
    lidar_dict = defaultdict(list)

    for topic, msg, t in bag.read_messages(topics=['/objects/obs1/rear/gps/rtkfix','/velodyne_points',
                '/objects/capture_vehicle/front/gps/rtkfix','/objects/capture_vehicle/rear/gps/rtkfix','/image_raw']):

        timestamp = msg.header.stamp.to_nsec()
        msgType = topicTypesMap[topic].msg_type
        if topic == '/velodyne_points':
            assert(msgType == 'sensor_msgs/PointCloud2')
            lidar2dict(msg, t, lidar_dict)
        elif topic == '/objects/obs1/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(timestamp, msg, obs_rear_rtk_dict)
        elif topic == '/objects/capture_vehicle/front/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(timestamp, msg, cap_front_rtk_dict)
        elif topic == '/objects/capture_vehicle/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(timestamp, msg, cap_rear_rtk_dict)
        elif topic == '/image_raw':
            assert(msgType == 'sensor_msgs/Image')
            cam_timestamps.append(t.to_nsec())


    obs_rear_rtk_df = pd.DataFrame(data=obs_rear_rtk_dict, columns=obs_rear_rtk_cols)
    cap_rear_rtk_df = pd.DataFrame(data=cap_rear_rtk_dict, columns=cap_rear_rtk_cols)
    cap_front_rtk_df = pd.DataFrame(data=cap_front_rtk_dict, columns=cap_front_rtk_cols)
    lidar_df = pd.DataFrame(data=lidar_dict, columns=lidar_cols)

    obs_rear_rtk_df.to_csv(os.path.join(outdir, 'obstacle_rear_rtk.csv'), index=False)
    cap_rear_rtk_df.to_csv(os.path.join(outdir, 'capture_vehicle_rear_rtk.csv'), index=False)
    cap_front_rtk_df.to_csv(os.path.join(outdir, 'capture_vehicle_front_rtk.csv'), index=False)
    lidar_df.to_csv(os.path.join(outdir,'./lidar_df.csv'), index=False)

    # obstale rear rtk, capture front/rear rtk data will be interpolated to lidar timestamp
    lidar_df['timestamp'] = pd.to_datetime(lidar_df['timestamp'])
    lidar_df.set_index(['timestamp'], inplace=True)
    lidar_df.index.rename('index', inplace=True)
    lidar_index_df = pd.DataFrame(index=lidar_df.index)

    obs_rear_rtk_interp = interpolate_to_camera(lidar_index_df, obs_rear_rtk_df)
    obs_rear_rtk_interp.to_csv(os.path.join(outdir, 'obstacle_rear_rtk_interp.csv'), header=True)
    obs_rear_rtk_interp_rec = obs_rear_rtk_interp.to_dict(orient='records')

    cap_rear_rtk_interp = interpolate_to_camera(lidar_index_df, cap_rear_rtk_df)
    cap_rear_rtk_interp.to_csv(os.path.join(outdir, 'capture_vehicle_rear_rtk_interp.csv'), header=True)
    cap_rear_rtk_interp_rec = cap_rear_rtk_interp.to_dict(orient='records')

    cap_front_rtk_interp = interpolate_to_camera(lidar_index_df, cap_front_rtk_df)
    cap_front_rtk_interp.to_csv(os.path.join(outdir, 'capture_vehicle_front_rtk_interp.csv'), header=True)
    cap_front_rtk_interp_rec = cap_front_rtk_interp.to_dict(orient='records')

    # transform coordinate system of obstacle from base gps to lidar position
    obs_poses_interp_transform = obstacle_coordinate_base2lidar(obs_rear_rtk_interp_rec, cap_rear_rtk_interp_rec, cap_front_rtk_interp_rec, mdr)

    thefile = open(os.path.join(outdir, 'obs_poses_interp_transformed.txt'), 'w')
    lidar_timestamps = lidar_df.index.astype('int')
    for cnt, item in enumerate(obs_poses_interp_transform):
        item["timestamp_lidar"] = lidar_timestamps[cnt]
        thefile.write("%s\n" % item)

    return obs_poses_interp_transform, cam_timestamps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Lidar Data Interpolate")
    parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROSbag name')
    parser.add_argument('--metadata', type=str, default="metadata.csv", help='metadata filename')
    parser.add_argument('--outdir', type=str, default=".", help='output directory')
    parser.add_argument('--trackletXML', type=str, default="tracklet_labels.xml", help='tracklet XML filename')
    args = parser.parse_args()
    dataset = args.dataset
    metadata = args.metadata
    outdir = args.outdir
    trackletFile = args.trackletXML

    try:
        f = open(dataset)
        f.close()

    except:
        print('Unable to read file: %s' % dataset)
        sys.exit()

    try:
        f = open(metadata)
        f.close()

    except:
        print('Unable to read file: %s' % metadata)
        sys.exit()


    f = open(trackletFile)
    data = f.read().replace('\n', '')
    xp = Parker()
    dataDict = xp.data(fromstring(data))
    f.close()

    obs_poses_interp_transform, cam_timestamps = interpolate_lidar_with_rtk(dataset, metadata, outdir)

    # test transformation.

    # tracklet_dict is the dictionary of object locations from tracklet_labels.xml
    tracklet_dict = clean_items_list(dataDict)
    put_timestamps_with_frame_ids(tracklet_dict, cam_timestamps)
    tracklet_df = pd.DataFrame(data=tracklet_dict,columns=["timestamp","tx","ty","tz","rx","ry","rz"])

    #thefile = open(os.path.join(outdir, 'tracklet_with_timestamps.txt'), 'w')
    #for item in tracklet_dict:
    #    thefile.write("%s\n" % item)

    # obs_poses_interp_transform is the dictionary of object locations obtained with transformation
    lidar_df = pd.DataFrame(data=obs_poses_interp_transform, columns=["timestamp_lidar","tx","ty","tz","rx","ry","rz"])
    lidar_df['timestamp_lidar'] = pd.to_datetime(lidar_df['timestamp_lidar'])
    lidar_df.set_index(['timestamp_lidar'], inplace=True)
    lidar_df.index.rename('index', inplace=True)

    # merge both dictionary and interpolate tracklet_dict object locations to lidar timestamps
    tracklet_interp2lidar = interpolate_to_camera(lidar_df, tracklet_df)
    tracklet_interp2lidar.to_csv(os.path.join(outdir, 'tracklet_interp2lidar.csv'), header=True)

    # merged dataframe has object location columns both from tracklet_xml and transformed object location.
    # compare them. difference should be small
    print(max(abs(tracklet_interp2lidar['tx_x']-tracklet_interp2lidar['tx_y'])))
    print(max(abs(tracklet_interp2lidar['ty_x']-tracklet_interp2lidar['ty_y'])))
    print(max(abs(tracklet_interp2lidar['tz_x']-tracklet_interp2lidar['tz_y'])))
