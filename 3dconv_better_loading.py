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
s = 8 
lin_s = 16*s*8*4
PIC_NUM = 15697
PIC_NUM_t = 15697
# PIC_NUM_t = 2000

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
            pics.append(img.transpose(2,1,0))
            i += 1
            if i == PIC_NUM_t:
    #TODO uncoment, just to speed things up
                break
    print("Pocet obrazku je: ", len(pics))
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
        if i  ==  PIC_NUM_t:
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
        # print("shape after convolution",xb.shape)
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
        self.pics =np.asarray( pics).transpose(0, 2, 1 , 3, 4)
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
    data, labels = load_data()
    # print(data)
    # print(data.shape)
    # print(data.shape)
    data_s = data.shape[0] - 11
    tmp_arr = []
    train_stacked = []
    for i in range(11):
        tmp_arr.append(0)
    for i in range(data_s):
        for j in range(11):
            tmp_arr[j] = data[i+j]
        stack = np.stack(tmp_arr, axis=0)
        train_stacked.append(stack)
    labels = labels[5:t_size-5]
    print("data shape", np.shape(data),"labels shape ", np.shape(labels))


    border = int(data.shape[0]*4/5)
    data_train, data_val = np.split(data, [border])
    labels_train, labels_val = np.split(labels, [border])
    # print("data train shape je", data_train.shape)
    
    val_stacked = []
    for i in range(v_num):
        for j in range(15):
            tmp_arr[j] = data_val[i+j]
        stack = np.stack(tmp_arr, axis=0)
        val_stacked.append(stack)
    labels_val = labels_val[7:t_size-7]

    # print(np.shape(train_stacked), np.shape(val_stacked))
    # print(labels_train.shape)
    dataset_tr = Dataset(train_stacked, labels_train)
    dataset_val = Dataset(val_stacked, labels_val)
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
        acc = 0
        processed = 0
        for i,batch in enumerate(train_dl):
            data = batch['rgbs']
            labels = batch['labels']
            key = batch['key']
            # print(data.shape)
            data = data.to(dev)
            labels = labels.to(dev)
            loss_batch(model, loss_fun, data, labels, opt)
            with torch.no_grad():
                value, num = loss_batch(model, loss_fun, data, labels)
                acc = (num*value + processed * acc)/(num + processed)
                processed += num

        print( " " )
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
    trn_loader, val_loader = get_loader(4, False)
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

    for example in val_loader:
        val_dat = example["rgbs"].to(dev)
        val_lab = example["labels"].to(dev)
        break
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
    res_before_training = []
    res_after_training = []
    with torch.no_grad():
        before_t = model(val_dat)
    fit(trn_loader, val_loader,model, opt, nn.MSELoss(), args.epochs)
    torch.save(model.state_dict(), "model")
    after_t = model(val_dat)
    print(before_t, after_t, val_lab)
    for i in range(len(val_lab.flatten())):
        print(before_t[i].item(), after_t[i].item(), val_lab[i].item())

if __name__ == "__main__":
    main()

