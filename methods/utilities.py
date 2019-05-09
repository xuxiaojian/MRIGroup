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


# Convert a python dict to markdown table
def dict_to_markdown_table(config_dict, name_section):
    info = '## ' + name_section + '\n'
    info = info + '|  Key  |  Value |\n|:----:|:---:|\n'

    for i in config_dict.keys():
        info = info + '|' + i + '|' + config_dict[i] + '|\n'

    info = info + '\n'

    return info
