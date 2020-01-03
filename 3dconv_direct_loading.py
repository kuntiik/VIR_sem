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
#path_pic = "audi/camera/camera/cam_front_center/"
path_labels = "labels/"
path_train_data = "3d_train/data"
path_train_labels = "3d_train/labels"
path_val_data = "3d_val/data"
path_val_label = "3d_val/labels"

s = 8 
lin_s = 16*s*8*4
PIC_NUM = 15697
PIC_NUM_t = 15697
# PIC_NUM_t = 2000

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class My_CNN(torch.nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()
        #super().__init__()
        self.conv1 = self._make_conv_layer(3,s)
        self.conv2 = self._make_conv_layer(s,2*s)
        self.conv3 = self._make_conv_layer(2*s, 4*s)
        self.conv4 = self._make_conv_layer(4*s, 8*s)
        self.conv_2d_1 = nn.Sequential(
            nn.Conv2d(4*s, 8*s, kernel_size = 3),
            nn.BatchNorm2d(8*s), nn.LeakyReLU(), nn.MaxPool2d((3,3)))
        self.conv_2d_2 = nn.Sequential(
            nn.Conv2d(8*s, 16*s, kernel_size = 3),
            nn.BatchNorm2d(16*s), nn.LeakyReLU(), nn.MaxPool2d((2,2)))
        self.fc1 = nn.Linear(lin_s, 512)
        self.fc2 = nn.Linear(512, 1)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, xb):
        #print("shape after convolution",xb.shape)
        xb = self.conv1(xb)
        # print("shape after convolution",xb.shape)
        xb = self.conv2(xb)
        # print("shape after convolution",xb.shape)
        xb = self.conv3(xb)
        # print("shape after convolution",xb.shape)
        xb = np.squeeze(xb, axis = 2)
        xb = self.conv_2d_1(xb) 
        # print("shape after convolution",xb.shape)
        xb = self.conv_2d_2(xb) 
        # print("shape after convolution",xb.shape)
        xb = xb.view(-1, lin_s)
        # print("shape after convolution",xb.shape)
        xb = F.relu(self.fc1(xb))
        xb = self.fc2(xb)
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
        return len(os.listdir(self.labels)) 

    def __getitem__(self, i):
        return{
#mby here should be transpose
            # 'pic':self.pics[i]/255,
            # 'label':self.labels[i],
            # 'key':i,
            'labels':np.load(self.labels + "/" + os.listdir(self.labels)[i])[0].astype('f4'),
            'rgbs':np.load(self.pics + "/" + os.listdir(self.pics)[i]).transpose(1,0,2,3).astype('f4')/255,
            'key':i,
        } 

def loss_batch(model, loss_function, data, labels, opt = None):
    # model_res = model(data)
    # print(model_res.shape)
    # print(labels.shape)
    loss = loss_function(model(data).flatten(), labels)
    # print("went thru model")
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(data)

def get_loader(bs = 8, opt = False):
    dataset_tr = Dataset(path_train_data, path_train_labels)
    dataset_val = Dataset(path_val_data, path_val_label)
    trn_loader = tdata.DataLoader(dataset_tr, batch_size = bs, shuffle = True)
    val_loader = tdata.DataLoader(dataset_val, batch_size = bs)
    return trn_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser('Simple MNIST classifier')
    parser.add_argument('--learning_rate', '-lr', default=0.00001, type=float)
    parser.add_argument('--epochs', '-e', default=30, type=int)
    parser.add_argument('--batch_size', '-bs', default=8, type=int)
    return parser.parse_args()

def fit(train_dl, val_dl, model, opt, loss_fun, epochs):
    for epoch in range(epochs):
        time1 = time.time()
        acc = 0
        processed = 0
        for i,batch in enumerate(train_dl):
            data = batch['rgbs']
            labels = batch['labels']
            key = batch['key']
            #data = data.transpose(0,2,1,3,4)
            data = data.to(dev)
            labels = labels.to(dev)
            model.train()
            loss_batch(model, loss_fun, data, labels, opt)
            model.eval()
            with torch.no_grad():
                value, num = loss_batch(model, loss_fun, data, labels)
                acc = (num*value + processed * acc)/(num + processed)
                processed += num

        print( " " )
        time2 = time.time() - time1
        hod = time2 // 3600
        minutes = (time2 % 3600) // 60
        print("Time to train one epoch is {}hod {}min {}s".format(hod, minutes, time2%60))
        print("Epoch: {}/{} \t Training set accuracy: {:.5f}".format(epoch+1, epochs, acc))

        evaluate(val_dl, model, epoch, epochs, loss_fun)

def evaluate(val_dl, model, epoch, epochs, loss_function):

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
            # if epoch > 10:
                # print(model(data).data, labels.data, value)
            acc = (num*value + processed * acc)/(num + processed)
            processed += num
    print("Epoch: {}/{} \t Validation set accuracy: {:.5f}".format(epoch+1, epochs, acc))
#just squared error averaged over the validation set



def  main():
    args = parse_args()
    # print(args)
    loss_fun = nn.MSELoss()
    trn_loader, val_loader = get_loader(8, False)
    # t_tst, v_tst = get_loader(1)
    # for example in t_tst:
        # tst = example["rgbs"].to(dev)
        # lab = example["labels"].to(dev)
        # break
    # testv_data = []
    # testv_labels = []
    # print(v_tst)
    # for example2 in v_tst:

        # tst_v = example2["rgbs"].to(dev)
        # lab_v = example2["labels"].to(dev)
        # testv_data.append(tst_v)
        # testv_labels.append(lab_v)

    # for example in val_loader:
        # p_val_data = example["labels"]
        # print (p_val_data.data)
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
    torch.save(model.state_dict(), "model")

if __name__ == "__main__":
    main()

