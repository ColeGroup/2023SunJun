import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import heapq
import random
from sklearn.preprocessing import MinMaxScaler



#超参数
len2=199
LR = 0.001
z=0.08
z1=z*1.4
z2=np.sqrt(np.square(z)+np.square(z1))

#距离 and 核函数
def kernel(x,y,l):
    dist = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))  #余弦距离
    up=np.square(dist)
    down=2*np.square(l)
    kx=np.exp(-(up/down)) #高斯核函数
    return kx

def kernel1(x,l):
    up=np.square(100*x)
    down=2*np.square(l)
    kx=np.exp(-(up/down)) #高斯核函数
    return kx
#sigmoid
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
#decoder


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32,output_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        decoder = self.sigmoid(out)
        return decoder

def get_col_types():
    trans_protocol = ['TCP', 'UDP']
    app_protocol = ['OTHER', 'HTTP', 'SSL',  'HTTPS', 'DNS','SAMBA','MYSQL']
    return trans_protocol,app_protocol



def anomaly__detection(file,white,black):
    Xs = pd.read_csv('controller/DEESM.csv')
    Xs = Xs.iloc[:, :24]
    Xs = np.array(Xs)
    Coder = Decoder(20, 24)
    loss_func = nn.MSELoss()
    Coder = torch.load('controller/DEESM.pkl')
    Coder.eval()
    data=pd.read_csv("static/%s"%file)
    trans_protocol, app_protocol = get_col_types()
    for index, row in data.iterrows():
        data.at[index, 'trans_protocol'] = trans_protocol.index(row[2])
        data.at[index, 'app_protocol'] = app_protocol.index(row[3])
    ip = data.iloc[:, :1]
    test_data = data.iloc[:,1:25]
    ip_np=np.array(ip)
    test_data_np=np.array(test_data)
    white_ip=white
    black_ip=black
    result=[]
    data_np=np.array(data)
    with torch.no_grad():
        for i in range(0, len(test_data)):
            a_d=data_np[i].tolist()
            if ip_np[i] in white_ip:
                a_d.extend('0')
                result.append(a_d)
                continue
            if ip_np[i] in black_ip:
                a_d.extend('1')
                result.append(a_d)
                continue
            gx = []
            max_gx = []
            for j in range(0, len2):
                gx.append(kernel(test_data_np[i], Xs[j], z))
            test_data1 = np.array(test_data_np[i])
            data_tensor = minmaxscaler(test_data1)
            data_tensor = torch.as_tensor(data_tensor.astype(float))
            data_tensor = data_tensor.float()
            #gx_tensor = np.array(gx)
            index = map(gx.index, heapq.nlargest(20, gx))
            index = sorted(list(index))
            for k in range(len(index)):
                max_gx.append(gx[index[k]])
            loss1 = np.mean(max_gx)
            max_gx = torch.as_tensor(max_gx)
            max_gx = max_gx.float()
            decoder = Coder(max_gx)
            loss = loss_func(decoder, data_tensor)
            if loss1 > 0.8 and loss < 0.1:
                a_d.extend('0')
            else:
                a_d.extend('1')
            result.append(a_d)
    return result





