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
from didiCompetition.tracklets.python.bag_to_kitti import get_obstacle_pos, estimate_obstacle_poses, rtk2dict, interpolate_to_camera


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
                '/objects/capture_vehicle/front/gps/rtkfix','/objects/capture_vehicle/rear/gps/rtkfix']):
            
        msgType = topicTypesMap[topic].msg_type
        if topic == '/velodyne_points':
            assert(msgType == 'sensor_msgs/PointCloud2')
            lidar2dict(msg, t, lidar_dict)
        elif topic == '/objects/obs1/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, obs_rear_rtk_dict) 
        elif topic == '/objects/capture_vehicle/front/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, cap_front_rtk_dict) 
        elif topic == '/objects/capture_vehicle/rear/gps/rtkfix':
            assert(msgType == 'nav_msgs/Odometry')
            rtk2dict(msg, cap_rear_rtk_dict) 
            
  
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
    obs_poses_interpolated = obstacle_coordinate_base2lidar(obs_rear_rtk_interp_rec, cap_rear_rtk_interp_rec, cap_front_rtk_interp_rec, mdr)
    
    thefile = open(os.path.join(outdir, 'obs_poses_interp_transformed.txt'), 'w')
    for item in obs_poses_interpolated:
        thefile.write("%s\n" % item)
    
    # update obstacle position wrt. lidar position
    #for ind, entry in enumerate(obs_poses):
    #    obs_rear_rtk_filtered.iloc[ind,2] = entry['tx']
    #    obs_rear_rtk_filtered.iloc[ind,3] = entry['ty']
    #    obs_rear_rtk_filtered.iloc[ind,4] = entry['tz']

    
      
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
