
import numpy as np 
import argparse 
import json
import time
from torch import nn
from torch import optim
import torch.utils.data as tdata
import torch.nn.functional as F
import os
import cv2
import torch
import torchvision
from shutil import copyfile

WIDTH = 1920
HEIGHT = 1208
path = "/local/temporary/audi/camera/"
path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
#path_pic = "audi/camera/camera/cam_front_center/"
path_labels = "labels/"
out_labels = "labels_selected/"
s = 32
lin_s = 256*12*7
PIC_NUM = 15697
PIC_NUM_t = 15697
#PIC_NUM_t = 100

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_frequency():
    freq = []
    json_freq=[]  #buffer, ktery nese informaci o jednotlivych obrazcich, do ktere skupiny uhlu patri
    for i in range(220):
        freq.append(0)
    for i in range(PIC_NUM):
        json_freq.append(0)
    counter=0
    for name in os.listdir(path_labels):
        f = open(path_labels + name, "rb")
        angl = json.load(f)['Angle']
        mag = int(abs(angl)//6)
        #if mag >= 60:
            #print(mag, angl)
        f.close()
        freq[mag] += 1
        json_freq[counter] = mag 
        # json_freq.append(mag)
        counter+=1
    return freq, json_freq

def load_data():
    i = 0 
    st1 = time.time()
    freq, json_freq=get_frequency()
    freq_checker=[]
    for l in range(220):
        freq_checker.append(0)


    labels = []
    for l in range(220):
        freq_checker[l] = 0
    for name in sorted(os.listdir(path_labels)):
        freq_checker[json_freq[i]] += 1
        if freq_checker[json_freq[i]] < 800:
            copyfile( path_labels + name, out_labels + name)
        i += 1
        if i  ==  PIC_NUM_t:
#TODO uncoment, just to speed things up
            break
    labels = np.asarray(labels)
    return labels

def main():
    load_data()

if __name__ == "__main__":
    main()