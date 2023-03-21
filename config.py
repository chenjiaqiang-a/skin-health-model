EXP_RUNNER = 'Chen'

TRAIN_CSV_PATH = './data/HX_Skin_Acne_Train_GroundTruth.csv'
TEST_CSV_PATH = './data/HX_Skin_Acne_Test_GroundTruth.csv'
IMAGE_DIR = './data/images'
DENSITY_MAP_DIR = './data/density_maps'

ACNE_CATEGORIES_TRUE = ['Clear', 'Almost', 'Mild',
                        'Mild to Moderate', 'Moderate',
                        'Moderate to Less Severe',
                        'Less Severe', 'Severe']
ACNE_CATEGORIES = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']

NUM_CLASSES = 8

DEVICE = 'cuda:0'
