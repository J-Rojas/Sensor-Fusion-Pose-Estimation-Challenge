#!/usr/bin/env python

import argparse
import cv2
import cv_bridge
from image_geometry import PinholeCameraModel
import tf
import rospy
import sys
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from scipy.optimize import minimize
import math
import random
import commentjson as json
import time
import yaml

settings = {}
cameraModel = PinholeCameraModel()
cameraInfo = CameraInfo()
camera = {}

def costFunction( x, sign = 1.0, debug = False ):
	global cameraModel, settings
	# ( tx, ty, tz, Ay, Ap, Ar )
	# format for static transform publisher
	parameters = x

	translation = [ parameters[ 0 ], parameters[ 1 ], parameters[ 2 ], 1.0 ]

	# euler_matrix( roll, pitch, yaw )
	rotationMatrix = tf.transformations.euler_matrix( parameters[ 5 ], parameters[ 4 ], parameters[ 3 ], axes=settings['axes'] )

	if debug:
		print( translation )
		print( rotationMatrix )

	rotationMatrix[ :, 3 ] = translation
	if debug:
		print( rotationMatrix )

	error = 0
	for i in range( 0, len( settings[ 'points' ] ) ):
		point = settings[ 'points' ][ i ]
		expectedUV = settings[ 'uvs' ][ i ]

		rotatedPoint = rotationMatrix.dot( point )
		if debug:
			print( rotatedPoint )

		uv = cameraModel.project3dToPixel( rotatedPoint )

		if debug:
			print( uv )

		diff = np.array( uv ) - np.array( expectedUV )
		error = error + math.sqrt( np.sum( diff ** 2 ) )
		if debug:
			print( diff )
			print( error )

	return sign * error

def cameraCalibration( cameraInfo ):
	global settings, cameraModel, camera

	cameraModel.fromCameraInfo( cameraInfo )

	print( 'Starting optimization' )
	result = minimize( costFunction, settings[ 'initialTransform' ], args = ( 1.0, False ), bounds = settings[ 'bounds' ], method = 'SLSQP', options = { 'disp': True, 'maxiter': 1000 } )

	while not result.success or result.fun > 30:
		for i in range( 0, len( settings[ 'initialTransform' ] ) ):
			settings[ 'initialTransform' ][ i ] = random.uniform( settings[ 'bounds' ][ i ][ 0 ], settings[ 'bounds' ][ i ][ 1 ] )
		print( '' )
		print( 'Trying new starting point:' )
		print( settings[ 'initialTransform' ] )
		result = minimize( costFunction, settings[ 'initialTransform' ], args = ( 1.0, False ), bounds = settings[ 'bounds' ], method = 'SLSQP', options = { 'disp': True, 'maxiter': 1000 } )

	print( 'Finished calibration' )
	print( '' )
	print( 'Final transform:' )
	print( result.x )
	print( 'Error: ' + str( result.fun ) )

	if settings[ 'inputFile' ]:
		print( 'Generating calibration image' )
		generateImage( result.x )

	sys.stdout.flush()
	rospy.signal_shutdown( 'Finished calibration' )

def generateImage( parameters ):
	global settings

	image = cv2.imread( settings[ 'inputFile' ] )

	# parameters = ( tx, ty, tz, Ay, Ap, Ar )
	# format for static transform publisher
	translation = [ parameters[ 0 ], parameters[ 1 ], parameters[ 2 ], 1.0 ]

	# euler_matrix( roll, pitch, yaw )
	rotationMatrix = tf.transformations.euler_matrix( parameters[ 5 ], parameters[ 4 ], parameters[ 3 ], axes=settings['axes'] )
	rotationMatrix[ :, 3 ] = translation

	for i in range( 0, len( settings[ 'points' ] ) ):
		point = settings[ 'points' ][ i ]
		expectedUV = settings[ 'uvs' ][ i ]

		rotatedPoint = rotationMatrix.dot( point )
		uv = cameraModel.project3dToPixel( rotatedPoint )

		cv2.circle( image, ( int( expectedUV[ 0 ] ), int( expectedUV[ 1 ] ) ), 5, cv2.cv.Scalar( 255, 0, 0 ), thickness = -1 )
		cv2.circle( image, ( int( uv[ 0 ] ), int( uv[ 1 ] ) ), 5, cv2.cv.Scalar( 0, 0, 255 ), thickness = -1 )

	cv2.imwrite( settings[ 'outputFile' ], image )

def main():
	global cameraModel, settings

	parser = argparse.ArgumentParser(description='Lidar/camera calibration')
	parser.add_argument('settings', type=str, default=None, help='Settings json file')
	parser.add_argument('camera_matrix', type=str, default=None, help='Camera Matrix YAML file')
	parser.add_argument('--input_img', type=str, default=None, help='Input image file')
	parser.add_argument('--output_img', type=str, default=None, help='Output image file')

	args = parser.parse_args()

	# load settings
	settingsFile = open(args.settings, 'r')
	settings = json.load( settingsFile )
	if args.input_img:
		settings[ 'inputFile' ] = args.input_img
		settings[ 'outputFile' ] = args.output_img

	# check settings
	if not settings[ 'bounds' ]:
		print( 'Parameter `bounds` not found in `' + sys.argv[ 1 ] + '`' )
		print( 'Using default bounds' )
		settings[ 'bounds' ] = [ [ -5, 5 ], [ -5, 5 ], [ -5, 5 ], [ 0, 2 * 3.14 ], [ 0, 2 * 3.14 ], [ 0, 2 * 3.14 ] ]

	if not settings[ 'initialTransform' ]:
		print( 'Parameter `initialTransform` not found in `' + sys.argv[ 1 ] + '`' )
		print( 'Using default starting point' )
		settings[ 'initialTransform' ] = [ 0, 0, 0, 0, 0, 0 ]

	if not settings[ 'points' ] or not settings[ 'uvs' ]:
		print( 'Point correspondences not found in `' + sys.argv[ 1 ] + '`' )
		print( 'Both `points` and `uvs` are required' )
		sys.exit( 2 )

	# load camera matrix file
	stream = file(args.camera_matrix, 'r')
	calib_data = yaml.load(stream)
	cam_info = CameraInfo()
	cam_info.width = calib_data['image_width']
	cam_info.height = calib_data['image_height']
	cam_info.K = calib_data['camera_matrix']['data']
	cam_info.D = calib_data['distortion_coefficients']['data']
	cam_info.R = calib_data['rectification_matrix']['data']
	cam_info.P = calib_data['projection_matrix']['data']
	cam_info.distortion_model = calib_data['distortion_model']

	# begin calibration
	cameraCalibration(cam_info)

if __name__ == '__main__':
	main()
