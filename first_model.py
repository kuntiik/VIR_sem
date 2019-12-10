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
        if i == 100:
#TODO uncoment, just to speed things up
            break
    pics = np.asarray(pics)
    elapsed1 = time.time() - st1
    print("time to get pictures: ",elapsed1, "s")
    i = 0
    st2 = time.time()
    for name in sorted(os.listdir(path_labels)):
        f = open(path_labels + name, "rb")
        labels.append(json.load(f)['Angle'])
        f.close()
        i += 1
        if i  ==  100:
#TODO uncoment, just to speed things up
            break
    labels = np.asarray(labels)
    elapsed2 = time.time() - st2
    print("time to get labels: ",elapsed2, "s")
    return pics, labels

class Dataset(tdata.Dataset):
    def __init__(self, pics, labels):
        super().__init__()
        self.pics = pics
        self.labels = labels

    def __len__(self):
        return self.pics.shape[0]

    def __getitem__(self, i):
        return{
#mby here should be transpose
            'pic':self.pics[i]/255,
            'label':self.labels[i],
            'key':i,
        } 

def get_loader(bs = 8):
    data, labels = load_data()
    border = int(data.shape[0]*4/5)
    data_train, data_val = np.split(data, [border])
    labels_train, labels_val = np.split(labels, [border])
    dataset_tr = Dataset(data_train, labels_train)
    dataset_val = Dataset(data_val, labels_val)
    trn_loader = tdata.DataLoader(dataset_tr, batch_size = bs, shuffle = True)
    val_loader = tdata.DataLoader(dataset_val, batch_size = bs*2)
    return trn_loader, val_loader
def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--learning_rate', '-lr', default=0.00001, type=float)
    parser.add_argument('--epochs', '-e', default=30, type=int)
    parser.add_argument('--batch_size', '-bs', default=8, type=int)


def  main():
    args = parse_args()
    trn_loader, val_loader = get_loader()

if __name__ == "__main__":
    main()

