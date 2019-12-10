import numpy as np 
import argparse 
from torch import nn
from torch import optim
import torch.utils.data as tdata
import os

path = "/local/temporary/audi/camera/"
path_pic = "/local/temporary/audi/camera/camera/cam_front_center/"
path_labels = "labels/

def load_data():
    pics = []
    labels = []
    for name in sorted(os.listdir(path_pic)):
        if name.endswith(".png"):
            pics.append(open(path_pic + name, "rb"))

    for name in sorted(os.listdir(path_labels)):
        f = open(path_labels + name, "rb")
        labels.append(json.load(f)['Angle'])

def main():
    load_data()

if __name__ == "__main__":
    main()

