from dataset import MRIData
from tools import copy_code, set_logging_file, psnr_tf, ssim_tf, save_predict
from method import unet3d, KerasCallBack, attention_unet3d
import tensorflow as tf
import os
import datetime
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

experiment_path = '/export/project/gan.weijie/experiment/'
train_folder = 'apr26-attention-unet3d-no-residual'
test_folder = 'apr24-unet3d-h0to7-liver_crop'

root_path = '/export/project/gan.weijie/dataset/mri_source/'
scan_lines = str(400)

train_mat_index = [0, 1, 2, 3, 4, 5, 6, 7]
valid_mat_index = [8]
test_mat_index = [9]

img_index = [16, 20, 22, 26, 49]
batch_size = 1
epoch_train = 100
epoch_save = 20
mode = 'train'
is_liver_crop = False
method = 'attention_unet3d'

method_dict = {
    'unet': unet3d,
    'attention_unet3d': attention_unet3d,
}

if mode == 'train':
    copy_code(src='/export/project/gan.weijie/tmp/pycharm_project_664/', dst=experiment_path + train_folder + '/')
    if not os.path.exists(experiment_path + train_folder + '/' + 'model/'):
        os.mkdir(experiment_path + train_folder + '/' + 'model/')
    set_logging_file(path=experiment_path + train_folder + '/')

    logging.root.info('[main] Load train data.')
    train_data = MRIData(root_path, scan_lines, train_mat_index, img_index, is_shuffle=True,
                         is_liver_crop=is_liver_crop)

    logging.root.info('[main] Load valid data.')
    valid_data = MRIData(root_path, scan_lines, valid_mat_index, img_index, is_shuffle=False, is_liver_crop=is_liver_crop)

    model = method_dict[method]([10, 320, 320, 1])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='mse',
                  metrics=[psnr_tf, ssim_tf])

    model.fit(train_data.tf_dataset.batch(batch_size).prefetch(batch_size), epochs=epoch_train,
              steps_per_epoch=train_data.data.batches() // batch_size,
              validation_data=valid_data.tf_dataset.batch(batch_size).prefetch(batch_size),
              validation_steps=valid_data.data.batches() // batch_size,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      filepath=experiment_path + train_folder + '/' + 'model/model.{epoch:02d}.h5',
                      period=epoch_save),
                  tf.keras.callbacks.ModelCheckpoint(
                      filepath=experiment_path + train_folder + '/' + 'model.best.psnr.h5',
                      save_best_only=True, monitor='val_psnr_tf', mode='max'),
                  KerasCallBack(experiment_path + train_folder + '/', train_data.img_dataset, valid_data.img_dataset),
              ])

else:
    if mode == 'test':
        set_logging_file(path=experiment_path + test_folder + '/')

        logging.root.info('[main] Load test data.')
        test_data = MRIData(root_path, scan_lines, test_mat_index, img_index, is_shuffle=False,
                            is_liver_crop=is_liver_crop)

        model = tf.keras.models.load_model(experiment_path + test_folder + '/model.best.psnr.h5',
                                           custom_objects={'psnr_tf': psnr_tf, 'ssim_tf': ssim_tf})

        test_evaluate = model.evaluate(test_data.tf_dataset.batch(batch_size).prefetch(batch_size), steps=test_data.data.batches() // batch_size,verbose=1)
        test_predict = model.predict(test_data.tf_dataset.batch(batch_size).prefetch(batch_size), steps=test_data.data.batches() // batch_size, verbose=1)

        save_predict(imgs=test_predict,
                     evaluation={'name': model.metrics_names, 'value': test_evaluate},
                     name=test_folder + ' + ' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + 'mat_index-' + str(test_mat_index) + '-model.best-',
                     path=experiment_path + test_folder + '/',
                     slim=True)
