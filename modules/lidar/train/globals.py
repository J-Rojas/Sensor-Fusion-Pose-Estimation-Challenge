BATCH_SIZE = 64
EPOCHS = 100
IMG_WIDTH = 1801
IMG_HEIGHT = 32
NUM_CHANNELS = 3
NUM_CLASSES = 2
NUM_REGRESSION_OUTPUTS = 24
K_NEGATIVE_SAMPLE_RATIO_WEIGHT = 4
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

PREDICTION_FILE_NAME = 'objects_obs1_lidar_predictions.csv'
PREDICTION_MD_FILE_NAME = 'objects_obs1_metadata.csv'

WEIGHT_BB = 0.01
LEARNING_RATE = 0.001

LIDAR_CONV_VERTICAL_STRIDE = 1

IMG_CAM_WIDTH = 1368
IMG_CAM_HEIGHT = 512
NUM_CAM_CHANNELS = 1

USE_FEATURE_WISE_BATCH_NORMALIZATION = True
USE_SAMPLE_WISE_BATCH_NORMALIZATION = False
