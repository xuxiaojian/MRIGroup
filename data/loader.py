from data.tools import patch, normalization
import numpy as np
import h5py
from skimage.util import view_as_windows


def mri_based(root_path, data_index):
    file = h5py.File(root_path + "mri_original.hdf5", "r")

    data = []
    label = []
    data_imgs = []
    label_imgs = []

    for i in data_index:
        print("[Info] data/loader.py: Reading %d Data" % i)
        file_index = "healthy/" + str(i) + "/data"
        data.append(file[file_index])

        file_index = "healthy/" + str(i) + "/label"
        label.append(file[file_index])

    return data, label, data_imgs, label_imgs


def original(root_path, data_index, patch_size, patch_step, is_patch=False, img_index=(10, 20)):

    data, label, data_imgs, label_imgs = mri_based(root_path, data_index)

    if is_patch:
        data_list = []
        label_list = []

        for i in range(data_index.__len__()):
            batches = data[i].shape[0]

            for batch in range(batches):
                data_temp = patch(data[i][batch], patch_size, patch_step)
                label_temp = patch(data[i][batch], patch_size, patch_step)

                data_temp = normalization(data_temp)
                label_temp = normalization(label_temp)

                data_list.append(data_temp)
                label_list.append(label_temp)

        data = np.expand_dims(np.concatenate(data_list, 0), -1)
        label = np.expand_dims(np.concatenate(label_list, 0), -1)

    else:
        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)

        for i in img_index:
            data_imgs.append(data[i])
            label_imgs.append(label[i])

        data_imgs = np.stack(data_imgs, axis=0)
        label_imgs = np.stack(label_imgs, axis=0)

        data = normalization(np.expand_dims(data, axis=-1))
        label = normalization(np.expand_dims(label, axis=-1))
        data_imgs = normalization(np.expand_dims(data_imgs, axis=-1))
        label_imgs = normalization(np.expand_dims(label_imgs, axis=-1))

    return data, label, data_imgs, label_imgs


def liver_crop(root_path, data_index, patch_size, patch_step, is_patch=False, img_index=(10, 20)):

    data, label, data_imgs, label_imgs = mri_based(root_path, data_index)

    if is_patch is not True:
        print("[Info] data/loader.py: is_patch should be True when type_ is liver_crop")
        return 0

    region = np.zeros([9, 4], dtype=np.int)  # Liver region
    region[0] = [116, 244, 70, 198]
    region[1] = [100, 220, 35, 195]
    region[2] = [175, 290, 105, 245]
    region[3] = [130, 240, 90, 240]
    region[4] = [90, 260, 75, 240]
    region[5] = [95, 245, 90, 245]
    region[6] = [90, 230, 95, 230]
    region[7] = [50, 180, 100, 235]
    region[8] = [60, 220, 70, 230]

    for i in range(data_index.__len__()):
        region_index = data_index[i] - 1
        data[i] = data[i][:, region[region_index, 0]:region[region_index, 1],
                          region[region_index, 2]:region[region_index, 3]]
        label[i] = label[i][:, region[region_index, 0]:region[region_index, 1],
                            region[region_index, 2]:region[region_index, 3]]

    for i in img_index:
        data_imgs.append(data[0][i])
        label_imgs.append(label[0][i])
    data_imgs = np.stack(data_imgs, axis=0)
    label_imgs = np.stack(label_imgs, axis=0)
    data_imgs = normalization(np.expand_dims(data_imgs, axis=-1))
    label_imgs = normalization(np.expand_dims(label_imgs, axis=-1))

    data_list = []
    label_list = []

    for i in range(data_index.__len__()):

        data_cur = data[i]
        label_cur = label[i]

        for batches in range(data_cur.shape[0]):

            data_temp = patch(data_cur[batches, :, :], patch_size, patch_step)
            label_temp = patch(label_cur[batches, :, :], patch_size, patch_step)

            data_temp= normalization(data_temp)
            label_temp = normalization(label_temp)

            data_list.append(data_temp)
            label_list.append(label_temp)

    data = np.expand_dims(np.concatenate(data_list, 0), -1)
    label = np.expand_dims(np.concatenate(label_list, 0), -1)

    return data, label, data_imgs, label_imgs


def original_structural_patch_3d(root_path, data_index, patch_size, patch_step, is_patch=False, img_index=(10, 20)):

    data, label, data_imgs, label_imgs = mri_based(root_path, data_index)

    for i in range(data_index.__len__()):
        data[i] = normalization(np.expand_dims(data[i], axis=-1))
        label[i] = normalization(np.expand_dims(label[i], axis=-1))

        data_temp = np.zeros(shape=[96, 10, 320, 320, 1])
        label_temp = np.zeros(shape=[96, 10, 320, 320, 1])

        for data_index in range(96):
            for patch_index in range(10):
                # print(data_index + 96*patch_index)
                data_temp[data_index, patch_index, :, :, :] = data[i][data_index + 96*patch_index, :, :, :]
                label_temp[data_index, patch_index, :, :, :] = label[i][data_index + 96 * patch_index, :, :, :]

        data[i] = data_temp
        label[i] = label_temp

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    for i in img_index:
        data_imgs.append(data[i])
        label_imgs.append(label[i])

    data_imgs = np.stack(data_imgs, axis=0)
    label_imgs = np.stack(label_imgs, axis=0)

    return data, label, data_imgs, label_imgs


def liver_crop_structural_patch_3d(root_path, data_index, patch_size, patch_step, is_patch=False, img_index=(10, 20)):

    data, label, data_imgs, label_imgs = mri_based(root_path, data_index)

    if is_patch is not True:
        print("[Info] data/loader.py: is_patch should be True when type_ is liver_crop")
        return 0

    region = np.zeros([9, 4], dtype=np.int)  # Liver region
    region[0] = [116, 244, 70, 198]
    region[1] = [100, 220, 35, 195]
    region[2] = [175, 290, 105, 245]
    region[3] = [130, 240, 90, 240]
    region[4] = [90, 260, 75, 240]
    region[5] = [95, 245, 90, 245]
    region[6] = [90, 230, 95, 230]
    region[7] = [50, 180, 100, 235]
    region[8] = [60, 220, 70, 230]

    for i in range(data_index.__len__()):
        region_index = data_index[i] - 1
        data[i] = data[i][:, region[region_index, 0]:region[region_index, 1],
                          region[region_index, 2]:region[region_index, 3]]
        label[i] = label[i][:, region[region_index, 0]:region[region_index, 1],
                            region[region_index, 2]:region[region_index, 3]]

    data_list = []
    label_list = []

    for i in range(data_index.__len__()):

        width = data[i].shape[1]
        height = data[i].shape[2]

        data_cur = np.zeros([96, 10, width, height])
        label_cur = np.zeros([96, 10, width, height])

        for m in range(96):
            for n in range(10):
                data_cur[m, n, :, :] = data[i][m + 96*n, :, :]
                label_cur[m, n, :, :] = label[i][m + 96 * n, :, :]

        for m in range(96):
            data_temp = data_cur[m]
            label_temp = label_cur[m]

            data_temp = view_as_windows(data_temp, [10, patch_size, patch_size], patch_step)
            label_temp = view_as_windows(label_temp, [10, patch_size, patch_size], patch_step)

            patch_nums = data_temp.shape[1] * data_temp.shape[2]
            data_temp = np.ascontiguousarray(data_temp)
            label_temp = np.ascontiguousarray(label_temp)

            data_temp.shape = [patch_nums, 10, patch_size, patch_size]
            label_temp.shape = [patch_nums, 10, patch_size, patch_size]

            data_temp = np.expand_dims(data_temp, -1)
            label_temp = np.expand_dims(label_temp, -1)

            for n in range(patch_nums):
                data_temp[n] = normalization(data_temp[n])
                label_temp[n] = normalization(label_temp[n])

            data_list.append(data_temp)
            label_list.append(label_temp)

    data = np.concatenate(data_list, 0)
    label = np.concatenate(label_list, 0)

    for i in img_index:
        data_imgs.append(data[i])
        label_imgs.append(label[i])

    data_imgs = np.stack(data_imgs, axis=0)
    label_imgs = np.stack(label_imgs, axis=0)

    return data, label, data_imgs, label_imgs
