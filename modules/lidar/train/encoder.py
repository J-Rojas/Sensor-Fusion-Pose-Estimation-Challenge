
# coding: utf-8

import numpy as np
import math
from process.extract_rosbag_lidar import X_MIN, Y_MIN, Y_MAX, RES_RAD
from keras.utils import to_categorical
from transforms3d._gohlketransforms import rotation_matrix

INPUT_SHAPE = (93, 1029, 2)

def project_2d(tx, ty, tz):
    d = np.sqrt(tx ** 2 + ty ** 2)    
    l2_norm = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    print('d={} l2={}'.format(d, l2_norm))
    
    x_img = np.arctan2(-ty, tx) / RES_RAD[1]
    y_img = np.arctan2(tz, d) / RES_RAD[0]
    
    # shift origin
    x_img -= X_MIN    
    y_img -= Y_MIN
        
    y_img = int(y_img)
    x_img = int(x_img)
    
    y_img = min(y_img, Y_MAX)
    y_img = max(y_img, 0)

    return (y_img, x_img)

#def generate_label(tx, ty, tz, gps_l, gps_w, gps_h):
def generate_label(tx, ty, tz, l, w, h):    
    bbox = []
    #bbox.append(project_2d(tx-gps_l, ty+gps_w, tz+gps_h))
    #bbox.append(project_2d(tx+gps_l, ty+gps_w, tz+gps_h))
    #bbox.append(project_2d(tx-gps_l, ty+gps_w, tz-gps_h))
    #bbox.append(project_2d(tx+gps_l, ty+gps_w, tz-gps_h))
    
    bbox.append(project_2d(tx-l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz+h/2.))
    bbox.append(project_2d(tx-l/2., ty+w/2., tz-h/2.))
    bbox.append(project_2d(tx+l/2., ty+w/2., tz-h/2.))
   
    bbox = np.array(bbox)
    print(bbox)
    
    upper_left_x = bbox.min(axis=0)[0]
    upper_left_y = bbox.min(axis=0)[1]
    lower_right_x = bbox.max(axis=0)[0]
    lower_right_y = bbox.max(axis=0)[1]
    print upper_left_x, upper_left_y, lower_right_x, lower_right_y
    
    label = np.ones(INPUT_SHAPE[:2])
    label[upper_left_x:lower_right_x, upper_left_y:lower_right_y] = 0  
    y = to_categorical(label, num_classes=2) #1st dimension: on-vehicle, 2nd dimension: off-vehicle
    
    return y

def main():
    #centroid of obstacle after interplation
    tx, ty, tz = (0.699597401296,-76.989,2.17780519741)
    #gps_l, gps_w, gps_h = (2.032, 0.7239, 1.6256)
    l, w, h = (4.2418,1.4478,1.5748)
    generate_label(tx, ty, tz, l, w, h)        
    
if __name__ == '__main__':
    main()



