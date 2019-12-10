import numpy as np 
import argparse 
from torch import nn
from torch import optim
import torch.utils.data as tdata
import os
import torchvision

path = "/local/temporary/audi/camera/"
path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
path_labels = "labels/

def load_data():
    pics = []
    labels = []
    train_dataset = torchvision.ImageFolder(
         root=path_pic, transform=torchvision.transforms.ToTensor())

    for name in sorted(os.listdir(path_labels)):
        f = open(path_labels + name, "rb")
        labels.append(json.load(f)['Angle'])
        f.close()

def main():
    load_data()

if __name__ == "__main__":
    main()

