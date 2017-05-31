#!/usr/bin/env python
import sys
import csv
sys.path.append('./lidar_module/')
import numpy as np
import argparse
import rospy
import tf
import cProfile
import StringIO
import pstats
import sensor_msgs.point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from lidar_module.pipeline import LIDARPipeline

frame_id = "lidar_pred_center"

lidar_pipeline = None

ENABLE_PROFILING = False

pr = cProfile.Profile() if ENABLE_PROFILING else None

def fake_model(points):
    return np.average(points, axis=1)
    

def add_frame(position):
    br = tf.TransformBroadcaster()
    print(position)
    br.sendTransform(tuple(position), (0,0,0,1), rospy.Time.now(), frame_id, 'velodyne')


def publish():
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()

    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    md = None
    for obs in metadata:
        if obs['obstacle_name'] == 'obs1':
            md = obs
    assert md, 'obs1 metadata not found'
    marker.scale.x = md['l']
    marker.scale.y = md['w']
    marker.scale.z = md['h']
    marker.color.r = 0.9
    marker.color.g = 0.9
    marker.color.b = 0.9
    marker.color.a = 0.9
    marker.lifetime = rospy.Duration()
    pub = rospy.Publisher("lidar_pred_center", Marker, queue_size=10)
    pub.publish(marker)
    

# render the tx, ty, tz out of the lidar cloud
def lidar_cloud_to_numpy(msg):
    if ENABLE_PROFILING:
        pr.enable()
    assert isinstance(msg, PointCloud2)
    points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
    points = np.array(list(points))
    #print(points)
    global lidar_pipeline
    if lidar_pipeline is None:
        lidar_pipeline = LIDARPipeline(args.weightsFile)
    position = lidar_pipeline.predict_position(points)
    if ENABLE_PROFILING:
        pr.disable()

    add_frame(position)
    publish()


def lidar_callback(msg, who):
    assert isinstance(msg, PointCloud2)
    lidar_cloud_to_numpy(msg)

def print_stats():
    if ENABLE_PROFILING:
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

def load_metadata(md_path):
    data = []
    with open(md_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # convert str to float
            row['l'] = float(row['l'])
            row['w'] = float(row['w'])
            row['h'] = float(row['h'])
            row['gps_l'] = float(row['gps_l'])
            row['gps_w'] = float(row['gps_w'])
            row['gps_h'] = float(row['gps_h'])
            data.append(row)
    return data

if __name__ == '__main__':

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Team-SF Ros Node')
    parser.add_argument('bag', type=str, default="", help='Model Filename')
    parser.add_argument('weightsFile', type=str, default="", help='Model Filename')
    parser.add_argument('metadataPath', type=str, default="", help='Metadata Path')
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    metadata = load_metadata(args.metadataPath)

    rospy.init_node('base_link_lidar_predict')
    rospy.on_shutdown(print_stats)
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, "")
    rospy.spin()