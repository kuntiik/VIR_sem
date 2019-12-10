import numpy as np 
import argparse 
import json
import time
from torch import nn
from torch import optim
import torch.utils.data as tdata
import os
import cv2
import torchvision

WIDTH = 1920
HEIGHT = 1208
path = "/local/temporary/audi/camera/"
# path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
path_pic = "audi/camera/camera/cam_front_center/"
path_labels = "labels/"

def load_data():
    pics = []
    labels = []
    im_w = 480
    im_h = 302
    i = 0 
    st1 = time.time()
    for name in sorted(os.listdir(path_pic)):
        img = cv2.imread(os.path.join(path_pic, name))
        img = cv2.resize(img, (im_w, im_h))
        pics.append(img)
        i += 1
        if i == 1000:
#TODO uncoment, just to speed things up
            break
            # print(i)
    elapsed1 = time.time() - st1
    print(elapsed1)
    i = 0
    st2 = time.time()
    for name in sorted(os.listdir(path_labels)):
        f = open(path_labels + name, "rb")
        labels.append(json.load(f)['Angle'])
        f.close()
        i += 1
        if i  ==  1000:
            print(i)
#TODO uncoment, just to speed things up
            break
    elapsed2 = time.time() - st2
    print(elapsed2)
    return pics, labels

class Dataset(tdata.Dataset):
    def __init__(self, pics, labels):
        super().__init__()
        self.pics = np.load(pics, mmap_mode='r')
        self.labels = np.load(labels, mmap_mode='r')

    def __len__(self):
        return self.pics.shape[0]

    def __getitem__(self, i):
        return{
#mby here should be transpose
            'pic':np.asarray(self.pics[i]).astype('f4')/255,
            'label':np.asarray(self.labels[i]).astype('f4'),
            'key':i,
                               }

def main():
    data, labels = load_data()
    dataset = Dataset(data, labels)
    print(dataset[1])

if __name__ == "__main__":
    main()

