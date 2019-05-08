import shutil
import logging
import os
import numpy as np


def copytree_code(src_path, dst_path):
    for i in range(100):  # 100 is picked manually : )
        path_ = dst_path + 'code_' + str(i) + '/'
        if not os.path.exists(path_):
            shutil.copytree(src=src_path, dst=path_)  # Copy and back current codes.
            return True


def new_folder(target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)


def set_logging(target_path):
    log_file = logging.FileHandler(filename=target_path + 'log.txt')
    log_file.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.root.addHandler(log_file)
    return True


def save_predict(predict, evaluation, target_name, target_path, is_slim=False, is_save_xy=False, x=None, y=None):
    import PIL
    import scipy.io as sio

    batches, depths, width, height, channel = predict.shape
    if is_slim:
        depths = 1

    imgs_list = []
    for depth in range(depths):
        for batch in range(batches):
            img_current = np.squeeze(predict[batch, depth, :, :, :])
            img_current -= np.amin(img_current)
            img_current /= np.amax(img_current)

            # noinspection PyUnresolvedReferences
            imgs_list.append(PIL.Image.fromarray(img_current))

    imgs_list[0].save(target_path + target_name + '.tiff', save_all=True, append_images=imgs_list[1:])

    f = open(target_path + target_name + '.txt', 'w')
    f.writelines(str(evaluation))
    f.close()

    if x is not None and y is not None and is_save_xy is True:
        data_dict = {'predict': predict, 'x': x, 'y': y}
    else:
        data_dict = {'predict': predict}

    sio.savemat(target_path + target_name + '.mat', data_dict)


# Convert a python dict to markdown table
def dict_to_markdown_table(config_dict, name_section):
    info = '## ' + name_section + '\n'
    info = info + '|  Key  |  Value |\n|:----:|:---:|\n'

    for i in config_dict.keys():
        info = info + '|' + i + '|' + config_dict[i] + '|\n'

    info = info + '\n'

    return info
