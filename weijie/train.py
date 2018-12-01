import configparser
import os
import sys

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Set Config File
config = configparser.ConfigParser()
config.read('./config.ini')

dataset_path = config['GLOBAL']['dataset_path']

if len(sys.argv) == 2 and str(sys.argv[1]) == 'unet':

    import unet
    from util.data_loader import read_matfiles, read_customed_vaild

    # Build Network
    net = unet.TFNetwork(img_channels=1, truth_channels=1, cost="mean_squared_error")

    # Read Data
    train_imgs = read_matfiles(index_read=[1], root_path=dataset_path)
    valid_imgs = read_customed_vaild(root_path=dataset_path)

    # SetUp Parameters
    model_path = config['PATH_UNET']['model_path']
    validation_path = config['PATH_UNET']['validation_path']
    batch_size = int(config['PAR_UNET']['batch_size'])

    # Begin Training
    trainer = unet.TFTrainer(net, batch_size=batch_size)
    trainer.train(train_imgs, model_path, valid_imgs, valid_size=6,  # valid_size=6 is predefined, be careful
                  training_iters=200, epochs=100, display_step=20, save_epoch=50, validation_path=validation_path)

else:

    print('[train.py]: Error - Wrong Argv.')
