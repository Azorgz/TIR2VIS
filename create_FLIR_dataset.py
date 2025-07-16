import shutil
import os
from os.path import isfile

root = os.getcwd() + '/'
# test Day vis
# path_data = "/home/godeta/Bureau/val/RGB/"
# # path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/val/RGB/"
# path_target = root + "datasets/FLIR/FLIR_datasets/testA/"
# f = open(root + "/img_list/FLIR/FLIR_testA_list.txt", "r")
# for x in f:
#     x = x.replace('\n', '').replace('png', 'jpg')
#     path_img = path_data + x
#     target_path = path_target + x
#     shutil.copy(path_img, path_target)
#
# # test Night ir
# path_data = "/home/godeta/Bureau/val/thermal_8_bit/"
# # path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_8_bit/"
# path_target = root + "datasets/FLIR/FLIR_datasets/testB/"
# f = open(root + "/img_list/FLIR/FLIR_testB_list.txt", "r")
# for x in f:
#     x = x.replace('\n', '').replace('png', 'jpeg')
#     path_img = path_data + x
#     target_path = path_target + x
#     shutil.copy(path_img, path_target)
#
# # train Day Vis
# path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/RGB/"
# path_target = root + "datasets/FLIR/FLIR_datasets/trainA/"
# f = open(root + "/img_list/FLIR/FLIR_trainA_list.txt", "r")
# for x in f:
#     x = x.replace('\n', '').replace('png', 'jpg')
#     path_img = path_data + x
#     target_path = path_target + x
#     shutil.copy(path_img, path_target)
#
# # train Night ir
# path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_8_bit/"
# path_target = root + "datasets/FLIR/FLIR_datasets/trainB/"
# f = open(root + "/img_list/FLIR/FLIR_trainB_list.txt", "r")
# for x in f:
#     x = x.replace('\n', '').replace('png', 'jpeg')
#     path_img = path_data + x
#     target_path = path_target + x
#     shutil.copy(path_img, path_target)

# # train Night vis
path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/RGB/"
path_target = root + "datasets/FLIR/FLIR_datasets/trainC_0/"
f = open(root + "/img_list/FLIR/FLIR_trainB_list.txt", "r")
for x in f:
    x = x.replace('\n', '').replace('png', 'jpg')
    path_img = path_data + x
    target_path = path_target + x
    if isfile(path_img):
        shutil.copy(path_img, path_target)

# Test Night vis
path_data = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/val/RGB/"
path_target = root + "datasets/FLIR/FLIR_datasets/testC_0/"
f = open(root + "/img_list/FLIR/FLIR_testB_list.txt", "r")
for x in f:
    x = x.replace('\n', '').replace('png', 'jpg')
    path_img = path_data + x
    target_path = path_target + x
    if isfile(path_img):
        shutil.copy(path_img, path_target)