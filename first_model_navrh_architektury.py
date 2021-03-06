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
path_labels = "labels/"
s = 16

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

class My_CNN(torch.nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()
        #super().__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels = s, kernel_size=3, stride=2, padding=0)
        conv2 = nn.Conv2d(in_channels=s, out_channels = 2*s, kernel_size=3, stride=2, padding=0)
        conv3 = nn.Conv2d(in_channels=2*s, out_channels = 4*s, kernel_size=3, stride=2, padding=0)
        conv4 = nn.Conv2d(in_channels=4*s, out_channels = 8*s, kernel_size=3, stride=2, padding=0)
        #self.mp = nn.Maxpool2d(kernel_size = 2)

        self.convs = torch.nn.Sequential(conv1, torch.nn.ReLU(), torch.nn.BatchNorm2d(s),torch.nn.MaxPool2d(kernel_size=3), \
            conv2, torch.nn.ReLU(), torch.nn.BatchNorm2d(2*s),torch.nn.MaxPool2d(kernel_size=3),conv3, torch.nn.ReLU()), torch.nn.BatchNorm2d(4*s), \
            conv4, torch.nn.ReLU()
        self.fc1 = nn.Linear(239*150, 1)
    def forward(self, xb):
        print(xb.shape)
        #xb = F.relu(self.conv1(xb))
        #xb = F.relu(self.conv1(xb))
        #xb = F.relu(self.conv2(xb))
        #xb = self.mp(xb)
        #xb = F.relu(self.conv3(xb))
        #xb = F.relu(self.conv4(xb))
        #xb = self.mp(xb)
        xb=self.convs(xb)
        print(xb.shape)
        xb = self.fc1(xb)
        return xb

    def weights_initialization(self):
        for layer in self.named_parameters():
            if type(layer) in {nn.Conv2d, nn.ConvTranspose2d}:
                nn.init.xavier_uniform_(layer.weight.data, gain=torch.sqrt(2))

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
            # 'pic':self.pics[i]/255,
            # 'label':self.labels[i],
            # 'key':i,
            'labels':np.asarray(self.labels[i]).astype('f4'),
            'rgbs':np.asarray(self.pics[i]).astype('f4')/255,
            'key':i,
        } 

def loss_batch(model, loss_function, data, labels, opt = None):
    loss = loss_function(model(data), labels)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(data)

def get_loader(bs = 8):
    data, labels = load_data()
    border = int(data.shape[0]*4/5)
    data_train, data_val = np.split(data, [border])
    labels_train, labels_val = np.split(labels, [border])
    dataset_tr = Dataset(data_train, labels_train)
    dataset_val = Dataset(data_val, labels_val)
    trn_loader = tdata.DataLoader(dataset_tr, batch_size = bs, shuffle = True)
    val_loader = tdata.DataLoader(dataset_val, batch_size = bs*2)
    # print(trn_loader.shape)
    return trn_loader, val_loader
def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--learning_rate', '-lr', default=0.00001, type=float)
    parser.add_argument('--epochs', '-e', default=30, type=int)
    parser.add_argument('--batch_size', '-bs', default=8, type=int)
    return parser.parse_args()

def fit(train_dl, val_dl, model, opt, loss_fun, epochs):
    for epoch in range(epochs):
        for batch in train_dl:
            data = batch['rgbs']
            labels = batch['labels']
            key = batch['key']
            print(data.shape)
            data = data.to(dev)
            label = data.to(dev)
            loss_batch(model, loss_fun, data, label, opt)

       # with torch.no_grad():
            #TODO evaluate loss in training
        
def evaluate(val_dl, model, epoch, loss_function):

    with torch.no_grad():
        acc = 0
        processed = 0 
        for batch in val_dl:
            data = batch['rgbs']
            labels = batch['labels']
            #key not needed yet
            key = batch['key']
            data = data.to(dev)
            labels = labels.to(dev)
            value, num = loss_batch(model, loss_function, data, labels)
            acc = (num*value + processed * acc)/(num + processed)
            processed += num
    print(acc)
#just squared error averaged over the validation set



def  main():
    args = parse_args()
    print(args)
    loss_fun = nn.MSELoss()
    trn_loader, val_loader = get_loader()
    model = My_CNN()
    #model_params = list(model.parameters())
    #model.weights_initialization()
    model = model.to(dev)
    # m_params = list(model.parameters())
    # opt = torch.optim.Adam(m_params, args.learning_rate)
    opt = torch.optim.Adam(model.parameters(), args.learning_rate)
    #opt = torch.optim.Adam(model_params, args.learning_rate)
    #opt=torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    fit(trn_loader, val_loader,model, opt, nn.MSELoss(), args.epochs)

if __name__ == "__main__":
    main()

