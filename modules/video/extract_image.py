

import numpy as np
import time
import os
from PIL import Image

'''
This method is for extracting the detected objects.
Which shall be used for training classifier for vehicle orientation detection
Invoke this method before drawing the overlay boxes on the image

'''

def extractObject(image, box, label,bagname,dir_name='extractedImages'):

    output_path = os.path.expanduser(dir_name)
    if not os.path.exists(output_path):
        print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    cbox = (left,top, right,bottom)
    crop = image.crop(cbox)

    base_filename = label + '_' + bagname + '_' + str(time.time())  # adding time to avoid overwriting
    filename = os.path.join(output_path, base_filename.replace(' ', '-') + '.jpg')

    return crop.save(filename)


