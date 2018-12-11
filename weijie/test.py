import configparser
import os
import sys
import numpy as np

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

# Set Global Variables
os.environ['CUDA_VISIBLE_DEVICES'] = config['GLOBAL']['index_gpu']
dataset_path = config['GLOBAL']['dataset_path']

if len(sys.argv) == 2 and str(sys.argv[1]) == 'scadec_unet':
    from util.dataLoader import ReadValidDataset, ReadRawDataset
    from net.scadec import unet

    # Set Parameter
    model_test_path = config['PAR_UNET']['model_test_path']
    test_path = config['PAR_UNET']['test_path']

    # Read Train and Valid Data
    mat_index_train = np.fromstring(string=config['PAR_UNET']['mat_index_train'], dtype=np.int, sep=',').tolist()
    data_index_valid = np.fromstring(string=config['PAR_UNET']['data_index_valid'], dtype=np.int, sep=',').tolist()

    train_imgs = ReadRawDataset(mat_index_train, dataset_path)
    valid_imgs = ReadValidDataset(data_index_valid, dataset_path)

    # Build Network
    net = unet.TFNetwork()
    # net.predict(model_test_path, test_path, valid_imgs)
    net.predict(model_test_path, test_path, train_imgs)
