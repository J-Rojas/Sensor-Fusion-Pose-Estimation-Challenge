import sys
sys.path.append('../')
import pyglet
import argparse
import numpy as np
import rosbag
import os
import matplotlib.image as mpimg
import sensor_msgs.point_cloud2
import pickle
from cv_bridge import CvBridge

from extract_rosbag_lidar import generate_lidar_2d_front_view
from extract_rosbag_lidar import save_lidar_2d_images
from common.birds_eye_view_generator import generate_birds_eye_view

class ROSBagExtractor:

    def __init__(self,
                 window_max_width=875,
                 topdown_res=.2,
                 cmap=None,
                 output_dir=None,
                 quiet=False):
        self.windows = {}
        self.bridge = CvBridge()
        self.window_max_width = window_max_width
        self.cmap = cmap
        self.output_dir = output_dir
        self.topdown_res = (topdown_res, topdown_res)
        self.quiet=quiet

        if output_dir is not None:
            if not(os.path.isdir(self.output_dir + '/lidar_360/')):
                os.makedirs(self.output_dir + '/lidar_360/')
            if not (os.path.isdir(self.output_dir + '/topdown/')):
                os.makedirs(self.output_dir + '/topdown/')
            if not (os.path.isdir(self.output_dir + '/camera/')):
                os.makedirs(self.output_dir + '/camera/')

    @staticmethod
    def save_images(output_dir, name, count, image):
        mpimg.imsave('./{}/{}_{}.png'.format(output_dir, name, count), image)

    @staticmethod
    def print_msg(msgType, topic, msg, time, startsec):
        t = time.to_sec()
        since_start = 0

        if 'sensor_msgs' in msgType or 'nav_msgs' in msgType:
            since_start = msg.header.stamp.to_sec() - startsec

        if msgType == 'sensor_msgs/PointCloud2':
            print(topic, msg.header.seq, since_start, 'nPoints=', msg.width * msg.height, t)

        elif msgType == 'sensor_msgs/NavSatFix':
            print(topic, msg.header.seq, since_start, msg.latitude, msg.longitude, msg.altitude, t)

        elif msgType == 'nav_msgs/Odometry':

            position = msg.pose.pose.position
            print(topic, msg.header.seq, since_start, position.x, position.y, position.z, t)

        elif msgType == 'sensor_msgs/Range':

            print(topic, msg.header.seq, since_start, msg.radiation_type, msg.field_of_view, msg.min_range, msg.max_range,
                  msg.range, t)

        elif msgType == 'sensor_msgs/Image':

            print(topic, msg.header.seq, msg.width, msg.height, since_start, t)

        elif msgType == 'sensor_msgs/CameraInfo':

            print(topic, msg.header.seq, since_start, msg.width, msg.height, msg.distortion_model, t)

        else:
            pass
            # print(topic, msg.header.seq, t-msg.header.stamp, msg, t)

    def get_window(self, topic, img):
        if self.windows.get(topic, None) is None:
            print(img.shape)
            ratio = self.window_max_width / float(img.shape[1])
            size = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
            self.windows[topic] = pyglet.window.Window(size[0], size[1], caption=topic)
        return self.windows[topic]

    @staticmethod
    def convert_img(img):
        return pyglet.image.ImageData(img.shape[1], img.shape[0], 'RGB', np.flipud(img).tobytes())

    def handle_msg(self, msg_type, topic, msg, timestamp, result):

        window = []
        img = []

        if msg_type in ['sensor_msgs/Image']:

            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            if not self.quiet:
                window.append(self.get_window(topic, cv_img))
            img.append(self.convert_img(cv_img))

            name = 'image'
            if 'center' in topic:
                name = 'center'
            elif 'left' in topic:
                name = 'left'
            elif 'right' in topic:
                name = 'right'

            if self.output_dir is not None:
                self.save_images(self.output_dir + '/camera/', name, timestamp, cv_img)

        elif msg_type in ['sensor_msgs/PointCloud2'] and 'velo' in topic:

            points = sensor_msgs.point_cloud2.read_points(msg, skip_nans=False)
            points = np.array(list(points))

            # render top down point cloud
            density_top_down = generate_birds_eye_view(points, timestamp, res=(.4,.4))
            density_top_down = density_top_down.astype(np.uint8)
            density_top_down = np.dstack((density_top_down, density_top_down, density_top_down))

            # render 360 view
            lidar_images = generate_lidar_2d_front_view(points, cmap=self.cmap)

            result['intensity'][str(timestamp)] = lidar_images['intensity']
            result['distance'][str(timestamp)] = lidar_images['distance']
            result['height'][str(timestamp)] = lidar_images['height']
            
            img.extend(
                map(self.convert_img, [
                    lidar_images['intensity'],
                    lidar_images['distance'],
                    lidar_images['height'],
                    density_top_down
                ])
            )

            # save files
            if self.output_dir is not None:
                save_lidar_2d_images(self.output_dir + '/lidar_360/', timestamp.to_nsec(), lidar_images)
                save_lidar_2d_images(self.output_dir + '/topdown/', timestamp.to_nsec(), {'density': density_top_down})

            if not self.quiet:
                window.extend([
                    self.get_window(topic + '/360/intensity', lidar_images['intensity']),
                    self.get_window(topic + '/360/distance', lidar_images['distance']),
                    self.get_window(topic + '/360/height', lidar_images['height']),
                    self.get_window(topic + '/topdown/density', density_top_down),
                ])

        if not self.quiet:
            for w, i in zip(window, img):
                w.switch_to()
                w.dispatch_events()
                size = w.get_size()
                i.blit(0, 0, width=size[0], height=size[1])
                w.flip()


def main():

    appTitle = "Udacity Team-SF: ROSbag viewer"
    parser = argparse.ArgumentParser(description=appTitle)
    parser.add_argument('bag_file', type=str, help='ROS Bag name')
    parser.add_argument('--skip', type=int, default="0", help='skip seconds')
    parser.add_argument('--topics', type=str, default=None, help='Topic list to display')
    parser.add_argument('--topdown_res', type=str, default=.2, help='Topdown image resolution')
    parser.add_argument('--lidar_cmap', type=str, default='jet', help='Colormap for lidar images (Default "jet")')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for images')
    parser.add_argument('--quiet', dest='quiet', action='store_true')
    parser.set_defaults(quiet=False)

    args = parser.parse_args()

    bag_file = args.bag_file
    output_dir = args.outdir

    if not os.path.isfile(bag_file):
        print('bag_file ' + bag_file + ' does not exist')
        sys.exit()

    if output_dir is not None and not(os.path.isdir(output_dir)):
        print('output_dir ' + output_dir + ' does not exist')
        sys.exit()

    skip = args.skip
    startsec = 0
    last_topic_time = {}
    maximum_gap_topic = {}
    topics_list = args.topics.split(',') if args.topics else None

    extractor = ROSBagExtractor(cmap=args.lidar_cmap,
                                output_dir=output_dir,
                                topdown_res=args.topdown_res,
                                quiet=args.quiet)

    print("reading rosbag ", bag_file)
    bag = rosbag.Bag(bag_file, 'r')
    topicTypesMap = bag.get_type_and_topic_info().topics

    result = {'intensity': {}, 'distance': {}, 'height': {}}
    for topic, msg, t in bag.read_messages(topics=topics_list):
        msgType = topicTypesMap[topic].msg_type
        if startsec == 0:
            startsec = t.to_sec()
            if skip < 24 * 60 * 60:
                skipping = t.to_sec() + skip
                print("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
            else:
                skipping = skip
                print("skipping to ", skip, " from ", startsec, " ...")
        else:
            if t.to_sec() > skipping:

                if last_topic_time.get(topic) != None:
                    gap = t.to_sec() - last_topic_time[topic]
                    if maximum_gap_topic.get(topic) == None or gap > maximum_gap_topic[topic]:
                        maximum_gap_topic[topic] = gap

                last_topic_time[topic] = t.to_sec()

                if not args.quiet:
                    extractor.print_msg(msgType, topic, msg, t, startsec)
                if not args.quiet and not output_dir:
                    extractor.handle_msg(msgType, topic, msg, t, result)
                    
    f = open(output_dir + '/lidar.p', 'wb')
    pickle.dump(result, f)
    f.close()
    #Load pickle: 
    #input  
    '''
    f = open(output_dir + '/lidar.p', 'rb')
    pickle_data = pickle.load(f)
    print(pickle_data)
    for mapType, pointMap in pickle_data.items():
        print(mapType)
        for t, value in pointMap.items():
            print(t)
            print(np.shape(value))
    '''
    
    #output
    '''
    distance
    1490149174663355139
    (93, 1029)
    ...
    intensity
    1490149174663355139
    (93, 1029)
    ...
    height
    1490149174663355139
    (93, 1029)
    ...
    
    '''
    print("Max interval between messages per topic")
    for key, value in sorted(maximum_gap_topic.iteritems(), key=lambda (k,v): (v,k)):
        print("    {}: {}".format(key, value))

# ***** main loop *****
if __name__ == "__main__":
    main()
