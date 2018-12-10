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

if len(sys.argv) == 2 and str(sys.argv[1]) == 'unet':

    from util.dataLoader import ReadValidDataset, ReadRawDataset
    from net import unet

    # Build Network
    net = unet.KesUNet(config)

    # Read Train and Valid Data
    mat_index_train = np.fromstring(string=config['PAR_UNET']['mat_index_train'], dtype=np.int, sep=',').tolist()
    data_index_valid = np.fromstring(string=config['PAR_UNET']['data_index_valid'], dtype=np.int, sep=',').tolist()

    train_imgs = ReadRawDataset(mat_index_train, dataset_path)
    valid_imgs = ReadValidDataset(data_index_valid, dataset_path)

    # Begin Train
    net.train(train_imgs, valid_imgs)

if len(sys.argv) == 2 and str(sys.argv[1]) == 'scadec_unet':

    from util.dataLoader import ReadValidDataset, ReadRawDataset
    from net.scadec import unet, imgs_util

    # Build Network
    net = unet.TFNetwork()

    # Read Train and Valid Data
    mat_index_train = np.fromstring(string=config['PAR_UNET']['mat_index_train'], dtype=np.int, sep=',').tolist()
    data_index_valid = np.fromstring(string=config['PAR_UNET']['data_index_valid'], dtype=np.int, sep=',').tolist()

    train_imgs = ReadRawDataset(mat_index_train, dataset_path)
    valid_imgs = ReadValidDataset(data_index_valid, dataset_path)

    train_imgs = imgs_util.SimpleDataProvider(train_imgs[0], train_imgs[1])
    valid_imgs = imgs_util.SimpleDataProvider(valid_imgs[0], valid_imgs[1])

    # Set Parameter
    model_path = config['PAR_UNET']['model_path']
    validation_path = config['PAR_UNET']['validation_path']

    batch_size = int(config['PAR_UNET']['batch_size'])
    epochs = int(config['PAR_UNET']['epochs'])
    epoch_save_model = int(config['PAR_UNET']['epoch_save_model'])

    # Train
    trainer = unet.TFTrainer(net, batch_size=batch_size)
    trainer.train(train_imgs, model_path, valid_imgs, 3, training_iters=200, epochs=epochs, save_epoch=epoch_save_model,
                  validation_path=validation_path)
