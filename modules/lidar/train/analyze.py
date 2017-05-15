import pandas as pd
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colors

def convert_to_polar(transforms):
    arr = []
    for item in transforms:
        x, y = np.float64(item['tx']), np.float64(item['ty'])
        arr.append(
            {
                'rho': np.sqrt(x**2+y**2),
                'phi': np.arctan2(y,x) * 180 / np.pi
            }
        )
    return arr

def main():

    parser = argparse.ArgumentParser(description='Lidar car/pedestrian analyzer')
    parser.add_argument("input_csv_file", type=str, default="../data/data_folders.csv", help="list of data folders for training")
    parser.add_argument("--dir_prefix", type=str, help="absolute path to folders")

    args = parser.parse_args()
    input_csv_file = args.input_csv_file
    dir_prefix = args.dir_prefix

    x = []
    y = []

    with open(input_csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:
            dir = row[0]
            interp_lidar_fname = dir_prefix+"/"+dir+"/obs_poses_interp_transform.csv"

            lidar_coord = []

            with open(interp_lidar_fname, 'r') as f:
                reader = csv.DictReader(f)
                # load lidar transformations
                for row in reader:
                    lidar_coord.append(row)

                # determine polar coordinates
                polar_coord = convert_to_polar(lidar_coord)

                # generate histogram
                x.extend(list(map(lambda x: x['rho'], polar_coord)))
                y.extend(list(map(lambda x: x['phi'], polar_coord)))

    hist = np.histogram2d(x, y, bins=[24, 90])
    norm=matplotlib.colors.LogNorm(vmin=1, vmax=1000, clip=True)
    plt.hist2d(y, x, bins=[90, 60], range=[[-180,180],[0,90]], norm=norm)
    plt.colorbar()
    plt.show()

main()
