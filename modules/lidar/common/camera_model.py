import tf
import yaml
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

class CameraModel:

    def __init__(self):
        self.camera_model = PinholeCameraModel()
        self.matrix = None

    def load_camera_calibration(self, camera_calibration_yaml, lidar_camera_calibration_yaml):

        stream = file(camera_calibration_yaml, 'r')
        calib_data = yaml.load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        stream.close()

        stream = file(lidar_camera_calibration_yaml, 'r')
        calib_data = yaml.load(stream)
        translation_data = calib_data['translation']['data']

        print(translation_data)

        translation = [translation_data[0], translation_data[1], translation_data[2], 1.0]
        rotation_data = calib_data['euler_rotations']['data']

        # euler_matrix( roll, pitch, yaw )
        rotationMatrix = tf.transformations.euler_matrix(rotation_data[2], rotation_data[1], rotation_data[0])
        rotationMatrix[:, 3] = translation

        self.camera_model.fromCameraInfo(cam_info)
        self.matrix = rotationMatrix

    def project_lidar_points_to_camera_2d(self, points):

        uv = []

        for point in points:
            rotatedPoint = self.matrix.dot(point)
            uv.append(self.camera_model.project3dToPixel(rotatedPoint))

        return uv


def generateImage(camera, centroid, inputFile, outputFile):
    import cv2

    image = cv2.imread(inputFile)

    uvs = camera.project_lidar_points_to_camera_2d([centroid])

    cv2.circle(image, (int(uvs[0][0]), int(uvs[0][1])), 5, cv2.cv.Scalar(0, 0, 255), thickness=-1)

    cv2.imwrite(outputFile, image)


def main():
    import argparse
    import tracket_parser
    import csv
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, help='Camera calibration yaml')
    parser.add_argument('--lidar', type=str, help='Lidar to Camera calibration yaml')
    parser.add_argument('--timestamps', type=str, help='Camera timestamps csv')
    parser.add_argument('--tracklet', type=str, help='Tracklet file')
    parser.add_argument('--input_dir', type=str, help='Rectified camera images directory')
    parser.add_argument('--output_dir', type=str, help='Annotation camera images directory')

    args = parser.parse_args()
    camera_calibration = args.camera
    lidar_camera_calibration = args.lidar
    camera_timestamps = args.timestamps
    tracklet = args.tracklet
    input_dir = args.input_dir
    output_dir = args.output_dir

    if camera_calibration is None or \
        lidar_camera_calibration is None or \
        camera_timestamps is None or \
        tracklet is None or \
        input_dir is None or \
        output_dir is None:
        parser.print_usage()
        sys.exit(-1)

    camera = CameraModel()
    camera.load_camera_calibration(camera_calibration, lidar_camera_calibration)

    data = None
    try:
        f = open(tracklet)
        data = f.read().replace('\n', '')
        f.close()

    except:
        print('Unable to read file: %s' % tracklet)
        f.close()
        exit(-1)

    timestamps = []
    try:
        f = open(camera_timestamps)
        csv_reader = csv.DictReader(f, delimiter=',', restval='')
        timestamps = []
        for row in csv_reader:
            timestamps.append(row['timestamp'])
        f.close()

    except:
        print('Unable to read file: %s' % camera_timestamps)
        f.close()
        exit(-1)

    dataDict = tracket_parser.xml_to_dict(data)
    cleaned = tracket_parser.clean_items_list(dataDict)
    tracket_parser.put_timestamps_with_frame_ids(cleaned, timestamps)

    for item in cleaned:
        ts = item['timestamp']
        generateImage(camera, [item['tx'], item['ty'], item['tz'], 1.0],
                      '{}/image_{}.png'.format(input_dir, ts),
                      '{}/image_{}.png'.format(output_dir, ts))

if __name__ == '__main__':
    main()
