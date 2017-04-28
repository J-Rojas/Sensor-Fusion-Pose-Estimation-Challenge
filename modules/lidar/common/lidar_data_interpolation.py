import sys
sys.path.append('../../')
import os
import json
import rosbag
import functools
import pandas as pd
from collections import defaultdict
import PyKDL as kd
import numpy as np
import argparse
from didiCompetition.tracklets.python.bag_to_kitti import get_obstacle_pos, estimate_obstacle_poses


# notes
# convert stamps to .to_nsec() in extract_rosbag.py while saving lidar images
#

#
# rtk to dictionary
#
def rtk2dict(msg, stamp, rtk_dict, rtk_type):
    rtk_dict["timestamp"].append(stamp.to_nsec())
    rtk_dict["tx"].append(msg.pose.pose.position.x)
    rtk_dict["ty"].append(msg.pose.pose.position.y)
    rtk_dict["tz"].append(msg.pose.pose.position.z)
   
    if (rtk_type == 'obs_rear'):
        rtk_dict["obs_rear_rtk"].append(rtk_type)
    elif (rtk_type == 'cap_rear'):
         rtk_dict["cap_rear_rtk"].append(rtk_type)
    elif (rtk_type == 'cap_front'):
        rtk_dict["cap_front_rtk"].append(rtk_type)
     
#
# lidar to dictionary
#
def lidar2dict(msg, stamp, lidar_dict):
    lidar_dict["timestamp"].append(stamp.to_nsec())
    lidar_dict["cap_lidar"].append("cap_lidar")
    #lidar_dict["timestamp_lidar"].append(stamp.to_nsec())

 
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


    metadata_df = pd.read_csv(metadata_filename, header=0, index_col=None, quotechar="'")
    mdr = metadata_df.to_dict(orient='records')

    bag = rosbag.Bag(bag_filename, 'r')
    topicTypesMap = bag.get_type_and_topic_info().topics
    
    obs_rear_rtk_cols = ["obs_rear_rtk", "timestamp", "tx", "ty", "tz"]
    obs_rear_rtk_dict = defaultdict(list)
    
    cap_rear_rtk_cols = ["cap_rear_rtk", "timestamp", "tx", "ty", "tz"]
    cap_rear_rtk_dict = defaultdict(list)
    
    cap_front_rtk_cols = ["cap_front_rtk", "timestamp", "tx", "ty", "tz"]
    cap_front_rtk_dict = defaultdict(list)
    
    lidar_cols = ["cap_lidar", "timestamp"]
    lidar_dict = defaultdict(list)
    
    for topic, msg, t in bag.read_messages(topics=['/objects/obs1/rear/gps/rtkfix','/velodyne_points',
                '/objects/capture_vehicle/front/gps/rtkfix','/objects/capture_vehicle/rear/gps/rtkfix']):
            
        msgType = topicTypesMap[topic].msg_type
        if topic == '/velodyne_points':
            assert(msgType == 'sensor_msgs/PointCloud2')
            lidar2dict(msg, t, lidar_dict)
        elif topic == '/objects/obs1/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, t, obs_rear_rtk_dict, 'obs_rear' ) 
        elif topic == '/objects/capture_vehicle/front/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, t, cap_front_rtk_dict, 'cap_front' ) 
        elif topic == '/objects/capture_vehicle/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, t, cap_rear_rtk_dict, 'cap_rear' ) 
            
  
    obs_rear_rtk_df = pd.DataFrame(data=obs_rear_rtk_dict, columns=obs_rear_rtk_cols)
    cap_rear_rtk_df = pd.DataFrame(data=cap_rear_rtk_dict, columns=cap_rear_rtk_cols)
    cap_front_rtk_df = pd.DataFrame(data=cap_front_rtk_dict, columns=cap_front_rtk_cols)  
    lidar_df = pd.DataFrame(data=lidar_dict, columns=lidar_cols)    
    #lidar_df.to_csv('./lidar_df.csv', header=True)
    
    # Camera dataframe needs to be indexed by timestamp for interpolation
    obs_rear_rtk_df['timestamp'] = pd.to_datetime(obs_rear_rtk_df['timestamp'])
    obs_rear_rtk_df.set_index(['timestamp'], inplace=True)
    obs_rear_rtk_df.index.rename('index', inplace=True)
    #obs_rear_rtk_df.to_csv('./obs_rear_rtk_df_datetime.csv', header=True)
 
    cap_rear_rtk_df['timestamp'] = pd.to_datetime(cap_rear_rtk_df['timestamp'])
    cap_rear_rtk_df.set_index(['timestamp'], inplace=True)
    cap_rear_rtk_df.index.rename('index', inplace=True)

    cap_front_rtk_df['timestamp'] = pd.to_datetime(cap_front_rtk_df['timestamp'])
    cap_front_rtk_df.set_index(['timestamp'], inplace=True)
    cap_front_rtk_df.index.rename('index', inplace=True)
    
    #lidar_df_timestamp = lidar_df['timestamp']
    lidar_df['timestamp'] = pd.to_datetime(lidar_df['timestamp'])
    lidar_df.set_index(['timestamp'], inplace=True)
    lidar_df.index.rename('index', inplace=True)
    #lidar_df.to_csv('./lidar_df_datetime.csv', header=True)
    
    #merged = functools.reduce(lambda left, right: pd.merge(
    #    left, right, how='outer', left_index=True, right_index=True), [lidar_df] + [obs_rear_rtk_df, cap_rear_rtk_df, cap_front_rtk_df])
    
    #filtered = merged.loc[lidar_df.index]  # back to only lidar rows
    #fname = outdir+'/lidar_interpolated.csv'
    #filtered.to_csv(fname, header=True)    
    
        
    obs_rear_rtk_merged = pd.merge(lidar_df, obs_rear_rtk_df, how='outer', left_index=True, right_index=True)
    obs_rear_rtk_merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')
    obs_rear_rtk_filtered = obs_rear_rtk_merged.loc[lidar_df.index]  # back to only lidar rows
    fname = outdir+'/obs_rear_rtk_filtered.csv'
    obs_rear_rtk_filtered.to_csv(fname, header=True)
    obs_rear_rtk_rec = obs_rear_rtk_filtered.to_dict(orient='records')
    
    cap_rear_rtk_merged = pd.merge(lidar_df, cap_rear_rtk_df, how='outer', left_index=True, right_index=True)
    cap_rear_rtk_merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')
    cap_rear_rtk_filtered = cap_rear_rtk_merged.loc[lidar_df.index]  # back to only lidar rows
    cap_rear_rtk_rec = cap_rear_rtk_filtered.to_dict(orient='records')

    cap_front_rtk_merged = pd.merge(lidar_df, cap_front_rtk_df, how='outer', left_index=True, right_index=True)
    cap_front_rtk_merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')
    cap_front_rtk_filtered = cap_front_rtk_merged.loc[lidar_df.index]  # back to only lidar rows
    cap_front_rtk_rec = cap_front_rtk_filtered.to_dict(orient='records')
  
    # transform coordinate system of obstacle from base gps to lidar position
    obs_poses = obstacle_coordinate_base2lidar(obs_rear_rtk_rec, cap_rear_rtk_rec, cap_front_rtk_rec, mdr)
    
    # update obstacle position wrt. lidar position
    for ind, entry in enumerate(obs_poses):
        obs_rear_rtk_filtered.iloc[ind,2] = entry['tx']
        obs_rear_rtk_filtered.iloc[ind,3] = entry['ty']
        obs_rear_rtk_filtered.iloc[ind,4] = entry['tz']

    fname = outdir+'/obs_rear_rtk_transformed.csv'
    obs_rear_rtk_filtered.to_csv(fname, header=True)
      
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('not enough argument')
        sys.exit()

    parser = argparse.ArgumentParser(description="Lidar Data Interpolate")
    parser.add_argument('--dataset', type=str, default="dataset.bag", help='Dataset/ROSbag name')
    parser.add_argument('--metadata', type=str, default="metadata.csv", help='metadata filename')
    parser.add_argument('--outdir', type=str, default=".", help='output directory')
    args = parser.parse_args()
    dataset = args.dataset
    metadata = args.metadata
    outdir = args.outdir

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

    interpolate_lidar_with_rtk(dataset, metadata, outdir)    
