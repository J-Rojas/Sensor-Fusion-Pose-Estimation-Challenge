import sys
import os
import json
import rosbag
import functools
import pandas as pd
from collections import defaultdict
import PyKDL as kd
import numpy as np
import argparse


# notes
# convert stamps to .to_nsec() in extract_rosbag.py while saving lidar images
#

#
# rtk to dictionary
#
def rtk2dict(msg, stamp, rtk_dict):
    rtk_dict["timestamp"].append(stamp.to_nsec())
    rtk_dict["tx"].append(msg.pose.pose.position.x)
    rtk_dict["ty"].append(msg.pose.pose.position.y)
    rtk_dict["tz"].append(msg.pose.pose.position.z)
    rtk_dict["type_rtk"].append("rtk")

#
# lidar to dictionary
#
def lidar2dict(msg, stamp, lidar_dict):
    lidar_dict["timestamp"].append(stamp.to_nsec())
    lidar_dict["type_lidar"].append("lidar")

    
#
# fill in obstacle position in lidar data with obstacle rtk data
#
def interpolate_lidar_with_rtk(bag_filename, outdir):

    bag = rosbag.Bag(bag_filename, 'r')
    topicTypesMap = bag.get_type_and_topic_info().topics
    
    obs_rear_rtk_cols = ["type_rtk", "timestamp", "tx", "ty", "tz"]
    obs_rear_rtk_dict = defaultdict(list)
    
    lidar_cols = ["type_lidar", "timestamp"]
    lidar_dict = defaultdict(list)
    
    for topic, msg, t in bag.read_messages(topics=['/objects/obs1/rear/gps/rtkfix','/velodyne_points']):
    
        #print(t.to_nsec())
        #print(msg.header.stamp)
        #print(msg.header.stamp.to_nsec())
        
        msgType = topicTypesMap[topic].msg_type
        if topic == '/velodyne_points':
            assert(msgType == 'sensor_msgs/PointCloud2')
            lidar2dict(msg, t, lidar_dict)
        elif topic == '/objects/obs1/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, t, obs_rear_rtk_dict ) 
 
    
    obs_rear_rtk_df = pd.DataFrame(data=obs_rear_rtk_dict, columns=obs_rear_rtk_cols)
    #obs_rear_rtk_df.to_csv('./obs_rear_rtk_df.csv', header=True)
    
    lidar_df = pd.DataFrame(data=lidar_dict, columns=lidar_cols)    
    #lidar_df.to_csv('./lidar_df.csv', header=True)
    
    # Camera dataframe needs to be indexed by timestamp for interpolation
    obs_rear_rtk_df['timestamp'] = pd.to_datetime(obs_rear_rtk_df['timestamp'])
    obs_rear_rtk_df.set_index(['timestamp'], inplace=True)
    obs_rear_rtk_df.index.rename('index', inplace=True)
    #obs_rear_rtk_df.to_csv('./obs_rear_rtk_df_datetime.csv', header=True)
    
    
    lidar_df['timestamp'] = pd.to_datetime(lidar_df['timestamp'])
    lidar_df.set_index(['timestamp'], inplace=True)
    lidar_df.index.rename('index', inplace=True)
    #lidar_df.to_csv('./lidar_df_datetime.csv', header=True)

    lidar_index_df = pd.DataFrame(index=lidar_df.index)
    
    #merged = functools.reduce(lambda left, right: pd.merge(
    #    left, right, how='outer', left_index=True, right_index=True), [rtk_df] + lidar_df)
        
    merged = pd.merge(lidar_df, obs_rear_rtk_df, how='outer', left_index=True, right_index=True)
    #merged.to_csv('./rtk_lidar_merged.csv', header=True)

    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')
    merged['timestamp'] = merged.index.astype('int') 
    #merged.to_csv('./rtk_lidar_merged_interpolated.csv', header=True)

    filtered = merged.loc[lidar_df.index]  # back to only lidar rows
    filtered['timestamp'] = filtered.index.astype('int') 
    
    fname = outdir+'/lidar_interpolated.csv'
    print(fname)
    filtered.to_csv(fname, header=True)    
    
    return filtered
    

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('not enough argument')
        sys.exit()

    parser = argparse.ArgumentParser(description="Lidar Data Interpolate")
    parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROS Bag name')
    parser.add_argument('--outdir', type=str, default=".", help='output directory')
    args = parser.parse_args()
    dataset = args.dataset
    outdir = args.outdir

    try:
        f = open(dataset)
        f.close()

    except:
        print('Unable to read file: %s' % dataset)
        sys.exit()
 
    interpolate_lidar_with_rtk(dataset, outdir)
     
