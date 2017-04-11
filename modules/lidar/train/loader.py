import numpy as np
import glob

BATCH_SIZE = 32

def parse_timestamp_from_file_name(filename):
    # TODO: get timestamp from file name
    timestamp = ?
    return timestamp

def load_tracklet_data_from_sources(list_source_dirs):

    ret_val = {}

    for d in list_source_dirs:
        # TODO: load the tracklet data files from each folder
        tracklet_data = ?
        ret_val[d] = tracklet_data

    return ret_val

def find_lidar_candidate_data_frames(lidar_files, list_tracklet_data, files):

    for data in list_tracklet_data:
        timestamp = data['timestamp']
        # TODO: determine the best matching lidar data frame file(s) for each timestamp
        # use the parse_timestamp_from_file_name function
        file = ...
        # append lidar candidate file to list
        files.append({'file': file, 'tracklet': data})

    return files


def aggregate_lidar_data_sources(list_source_dirs, tracklet_data_dict):

    files = []

    for d in list_source_dirs:
        lidar_files = glob.glob("{}/*.png".format(d))
        find_lidar_candidate_data_frames(lidar_files, tracklet_data_dict[d], files)

    return files

def aggregate_data(csv_sources):

    # determine list of data sources to load
    list_source_dirs = loadCSV(csv_sources)

    tracklet_data_dict = load_tracklet_data_from_sources(list_source_dirs)

    return aggregate_lidar_data_sources(list_source_dirs, tracklet_data_dict)

def generate_tensors(lidar_data_sources):

    # TODO: shuffle the data
    X = []
    y = []

    for data in lidar_data_sources:
        # TODO: Generate X tensor
        lidar_file = data['file']
        x_vec = ...


        # TODO: Generate y tensor
        tracklet = data['tracklet']
        y_vec = ...

        X.append(x_vec)
        y.append(y_vec)

        if len(X) == BATCH_SIZE:
            yield (np.array(X), np.array(y))
            X = []
            y = []

    yield (np.array(X), np.array(y))

def main():

    # TODO: do tests
    pass

# ***** main loop *****
if __name__ == "__main__":
    main()