import numpy as np
import glob
import argparse
import csv
import sys
import random
import pickle

from collections import defaultdict

BATCH_SIZE = 32
IMG_WIDTH = 1029
IMG_HEIGHT = 93

def usage():
    print('Loads training data with ground truths and generate training batches')
    print('Usage: python loader.py --input_csv_file [csv file of data folders]')

    
#
# read in images/ground truths batch by batch 
#
def data_generator(tx, ty, tz, pickle_dir_and_prefix):

	image_distace = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH), dtype=float)
	image_height = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH), dtype=float)	
	image_intensity = np.ndarray(shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH), dtype=float)		
	obj_location = np.ndarray(shape=(BATCH_SIZE,3), dtype=float)
	
	batch_index = 0
	size = len(tx)
	num_batches = size/BATCH_SIZE
	size = num_batches*BATCH_SIZE
	

	while 1:
		      
		for ind in range(size):
		
		    fname = pickle_dir_and_prefix[ind]+"_distance_float.lidar.p"
		    f = open(fname, 'rb')
		    pickle_data = pickle.load(f)
		    img_arr = np.asarray(pickle_data, dtype='float32')
		    np.copyto(image_distace[batch_index],img_arr)
		    f.close();
			
		    fname = pickle_dir_and_prefix[ind]+"_height_float.lidar.p"
		    f = open(fname, 'rb')
		    pickle_data = pickle.load(f)
		    img_arr = np.asarray(pickle_data, dtype='float32')
		    np.copyto(image_height[batch_index],img_arr)
		    f.close();
			
		    fname = pickle_dir_and_prefix[ind]+"_intensity_float.lidar.p"
		    f = open(fname, 'rb')
		    pickle_data = pickle.load(f)
		    img_arr = np.asarray(pickle_data, dtype='float32')
		    np.copyto(image_intensity[batch_index],img_arr)
		    f.close();
		    
		    obj_location[batch_index][0] = tx[ind]
		    obj_location[batch_index][1] = ty[ind]
		    obj_location[batch_index][2] = tz[ind]
		    
		    batch_index = batch_index + 1
		    
		    if (batch_index >= BATCH_SIZE):
		        batch_index = 0
		        yield (image_distace, image_height, image_intensity, obj_location)


        ziplist = list(zip(tx, ty, tz, img_dir_and_prefix))
        random.shuffle(ziplist)
        tx, ty, tz, img_dir_and_prefix = zip(*ziplist)


#
# read input csv file to get the list of directories
#
def get_data_and_ground_truth(csv_sources):
    
    txl = []
    tyl = []
    tzl = []
    pickle_dir_and_prefix = []

    with open(csv_sources) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
     
        for row in readCSV:
            dir = row[0]
            interp_lidar_fname = dir+"/lidar_interpolated.csv"
            
            with open(interp_lidar_fname) as csvfile_2:
                readCSV_2 = csv.DictReader(csvfile_2, delimiter=',')
                
                for row2 in readCSV_2:
                    ts = row2['timestamp']
                    tx = row2['tx']
                    ty = row2['ty']
                    tz = row2['tz']
                    
                    pickle_dir_prefix = dir+"/lidar_360/"+ts
                    pickle_dir_and_prefix.append(pickle_dir_prefix)
                    txl.append(tx)
                    tyl.append(ty)
                    tzl.append(tz)
                    
                    
    return txl,tyl,tzl,pickle_dir_and_prefix



# ***** main loop *****
if __name__ == "__main__":

    if len(sys.argv) < 2:
        usage()
        sys.exit()
 
    parser = argparse.ArgumentParser(description="Load training data and ground truths")
    parser.add_argument("--input_csv_file", type=str, default="data_folders.csv", help="data folder .csv")


    args = parser.parse_args()
    input_csv_file = args.input_csv_file

    
    try:
        f = open(input_csv_file)
        f.close()
    except:
        print('Unable to read file: %s' % input_csv_file)
        f.close()
        sys.exit()

    # determine list of data sources and ground truths to load
    tx,ty,tz,pickle_dir_and_prefix = get_data_and_ground_truth(input_csv_file)
   
    # generate data in batches
    generator = data_generator(tx, ty, tz, pickle_dir_and_prefix)
    image_distace, image_height, image_intensity, obj_location = next(generator)
    print(image_intensity)
    
    
    
    
    
    
    
