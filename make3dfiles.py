import numpy as np 
import argparse 
import json
import random
import time
from torch import nn
from torch import optim
import torch.utils.data as tdata
import torch.nn.functional as F
import os
import cv2
import torch
import torchvision
import shutil

WIDTH = 1920
HEIGHT = 1208
path = "/local/temporary/audi/camera/"
#path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
path_pic = "pics_all"
#path_pic = "audi/camera/camera/cam_front_center/"
path_labels = "labels/"
s = 8 
lin_s = 16*s*8*4
frame_size = 11
# PIC_NUM_t = 2000
trn_folder = '3d_train'
val_folder = '3d_val'

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# shutil.rmtree(trn_folder)
# shutil.rmtree(val_folder)

os.mkdir(trn_folder)
os.mkdir(val_folder)
os.mkdir(trn_folder + "/data")
os.mkdir(trn_folder + "/labels")
os.mkdir(val_folder + "/data")
os.mkdir(val_folder + "/labels")

pic_num = len(os.listdir(path_pic))

i = 0
train_count = 0
val_count = 0
tmp_arr = []
label_value = []
label_value.append(0)
for k in range(frame_size):
    tmp_arr.append(0)

while i < (pic_num - frame_size + 1):


    for j in range(frame_size):
        tmp_arr[j] = cv2.imread(path_pic+ "/" + sorted(os.listdir(path_pic))[i+j]).transpose(2,1,0)
    stack = np.stack(tmp_arr, axis=0)
    np.shape(stack)

    lab_name =  sorted(os.listdir(path_labels))[int(i + (frame_size +  1)/2)]
    f = open(path_labels + "/" + lab_name, "rb")
    label_value[0] = json.load(f)['Angle']
    f.close()
    label_value = np.asarray(label_value)
    
    decision = random.uniform(0,10)
    if decision < 8:
        np.save(trn_folder +  "/data/" + str(train_count) , stack)
        np.save(trn_folder + "/labels/" + str(train_count) , label_value)
        train_count +=1
    else:
        np.save(val_folder +  "/data/" + str(val_count) , stack)
        np.save(val_folder + "/labels/" + str(val_count) , label_value)
        val_count += 1
    
    i += 1




