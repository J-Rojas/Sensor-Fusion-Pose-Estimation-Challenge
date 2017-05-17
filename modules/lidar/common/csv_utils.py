import csv

class DirSet:

    def __init__(self):
        self.dir = ""
        self.mdr = {}

def foreach_dirset(input_csv_file, dir_prefix, doFunc):
    with open(input_csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:

            dirset = DirSet()
            dirset.dir = dir_prefix+"/"+row[0]

            metadata_file_name = row[1]

            with open(dir_prefix+"/"+metadata_file_name) as metafile:
                records = csv.DictReader(metafile)
                mdr = []
                for record in records:
                    mdr.append(record)
                dirset.mdr = mdr[0]

            doFunc(dirset)

def load_lidar_interp(dir):

    lidar_coord = []
    interp_lidar_fname = dir+"/obs_poses_interp_transform.csv"
    with open(interp_lidar_fname, 'r') as f:
        reader = csv.DictReader(f)
        # load lidar transformations
        for row in reader:
            lidar_coord.append(row)

    return lidar_coord
