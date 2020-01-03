
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

WIDTH = 1920
HEIGHT = 1208
path = "/local/temporary/audi/camera/"
path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
#path_pic = "audi/camera/camera/cam_front_center/"
pic_out = "pics_all/"
path_labels = "labels/"
s = 32
lin_s = 256*12*7
PIC_NUM = 15697
PIC_NUM_t = 15697
#PIC_NUM_t = 100

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_data():
    pics = []
    labels = []
    im_w = 480
    im_h = 302
    i = 0 
    st1 = time.time()

    for name in sorted(os.listdir(path_pic)):
        if name.endswith('.png'):
            img = cv2.imread(os.path.join(path_pic, name))
            img = cv2.resize(img, (im_w, im_h))
            cv2.imwrite(pic_out + name, img)
            pics.append(img.transpose(2,1,0))
    print("Pocet obrazku je: ", len(pics))
    pics = np.asarray(pics)
    elapsed1 = time.time() - st1
    print("time to get pictures: ",elapsed1, "s")
    return pics
def main():
    load_data()

if __name__ == "__main__":
    main()
