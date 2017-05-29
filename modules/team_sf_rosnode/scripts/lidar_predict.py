#!/usr/bin/env python
import sys
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
    marker.scale.x = 15
    marker.scale.y = 15
    marker.scale.z = 15
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


if __name__ == '__main__':

    print(sys.argv)

    parser = argparse.ArgumentParser(description='Team-SF Ros Node')
    parser.add_argument('bag', type=str, default="", help='Model Filename')
    parser.add_argument('weightsFile', type=str, default="", help='Model Filename')
    parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    rospy.init_node('base_link_lidar_predict')
    rospy.on_shutdown(print_stats)
    rospy.Subscriber('/velodyne_points', PointCloud2, lidar_callback, "")
    rospy.spin()