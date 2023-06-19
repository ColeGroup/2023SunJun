import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import heapq
import random
import time
from torch.autograd import Variable

#训练数据加载
delete=pd.read_csv("delete_normal_kdd.csv")
train_data=delete.iloc[:,:12]
train_label=delete.iloc[:,12]
train_data = np.array(train_data)
train_label = np.array(train_label)

#esl训练数据
#Xs=pd.read_csv("dasesm_new3.csv")
Xs=pd.read_csv("CESDDM.csv")
Xs=Xs.iloc[:,:12]
Xs = np.array(Xs)

#测试数据加载
test1=pd.read_csv("test.csv")
test_data=test1.iloc[:,:12]
test_label=test1.iloc[:,12]
test_data=np.array(test_data)
test_label=np.array(test_label)

#超参数
EPOCH=15
len1=20000#len(train_data)
len2=200
len3=len(test_data)
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
#fx
def fx_(x):
    fx = 0
    for i in range(len1):
        fx=fx+kernel(x,train_data[i],z)
    fx=fx/len1
    return fx

#px
def px_(x):
    px=0
    for i in range(len2):
        px=px+kernel(x,Xs[i],z)
    px=px/len2
    return px

#原始的Xs
def or_Xs():
    Xs=[]
    for i in range(len2):
        Xs.append(train_data[i])
    Xs = np.array(Xs)
    return Xs
Xs=or_Xs()

#初始fx
def or_fx():
    fx=[]
    for i in range(len2):
        a=fx_(Xs[i])
        fx.append(a)
    return fx

#初始px
def or_px():
    px=[]
    for i in range(len2):
        a=px_(Xs[i])
        px.append(a)
    return px
fx=or_fx()
px=or_px()

#替换谁
def max_re(x,y):
    maxj = []
    for i in range(len2):
        maxj.append(x[i] - y[i])
    j = maxj.index(max(maxj))
    return j

#是否替换
def max_r(x,y):
    up=fx[x]*px_(y)
    down=px[x]*fx_(y)
    if down==0:
        a=1
    else:
        a=up/down
    a=1-a
    c=max(a,0)
    return c

#esl_loss
def esl_loss():
    loss=0
    for i in range(len2):
        a=fx[i]/px[i]
        loss+=a
    loss=loss/len2
    return loss

#decoder
class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        #self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 24)
        self.linear4 = nn.Linear(24,output_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        #out = self.linear2(out)
        #out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        decoder = self.sigmoid(out)
        return decoder

#归一化
def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)

#训练
device = torch.device('cuda')
Coder= Decoder(20,12).to(device)
print(Coder)
optimizer = torch.optim.Adam(Coder.parameters(),lr=LR)
loss_func = nn.MSELoss()
t = 1
for epoch in range(EPOCH):
    epoch_loss = []
    p=0
    for i in range(len2, len1):
        #start=time.time()
        gx = []
        max_gx=[]
        for j in range(len2):
            gx.append(kernel(train_data[i], Xs[j], z))
        train_data1=np.array(train_data[i])
        data_tensor=minmaxscaler(train_data1)
        data_tensor =torch.tensor(data_tensor,requires_grad=True)
        data_tensor = data_tensor.float().to(device)
        gx_tensor=np.array(gx)
        #max_gx = heapq.nlargest(10, gx_tensor)
        index =map(gx.index, heapq.nlargest(20, gx))
        #index1 = sorted(list(index1))
        #index = map(gx.index, heapq.nsmallest(20,heapq.nlargest(30, gx)))
        index = sorted(list(index))
        for k in range(len(index)):
            max_gx.append(gx[index[k]])
        max_gx1=torch.tensor(max_gx,requires_grad=True)
        max_gx1=max_gx1.float().to(device)
        decoder = Coder(max_gx1)
        loss = loss_func(decoder, data_tensor)
        optimizer.zero_grad()
        max_gx1.retain_grad()
        loss.backward()
        gxd=(max_gx1.grad).tolist()
        max_gxd=list(filter(lambda x: x > 0, gxd))
        xs_max=[gx_tensor.tolist().index(i) for i in [max_gx[i].tolist()  for i in [gxd.index(i) for i in list(filter(lambda x: x > 0, gxd))]]]
        min_gxd=list(filter(lambda x: x<0,gxd))
        xs_min=[gx_tensor.tolist().index(i) for i in [max_gx[i].tolist()  for i in [gxd.index(i) for i in list(filter(lambda x: x < 0, gxd))]]]
        optimizer.step()
        if t>3000:
            t=3000
        a = (1.2 * t) / ((1.2 * t) + 1)
        b = 1 - a
        for k in range(len(xs_max)):
            #fx[xs_max[k]] = (a * fx[xs_max[k]]) + (b * kernel1(max_gxd[k], z))
            fx[xs_max[k]] = (a * fx[xs_max[k]]) + (b * 0.1*sigmoid(100*max_gxd[k]))
            #a1=fx[xs_max[k]]
            #a2=px[xs_max[k]]
        for k in range(len(xs_min)):
            #fx[xs_min[k]] = (a * fx[xs_min[k]]) - (b * kernel1(min_gxd[k], z))
            fx[xs_min[k]] = (a * fx[xs_min[k]]) - (b * 0.1*sigmoid(100*min_gxd[k]))
            if fx[xs_min[k]]<0:
                fx[xs_min[k]]=0
            #a1=fx[xs_min[k]]
            #a2=px[xs_min[k]]
        for k in range(len2):
            if k not in xs_max:
                if k not in xs_min:
                    fx[k]=a*fx[k]
                else:
                    continue
            else:
                continue
        re = max_re(px, fx)
        r = random.random()
        n=max_r(re,train_data1)
        if r<n:
            Xs[re]=train_data1
            fx[re]=fx_(Xs[re])
            px[re]=px_(Xs[re])
            p=p+1
            eloss = esl_loss()
            print(eloss)
        epoch_loss.append(loss.item())
        t = t + 1
    print(p)
    print("epoch {}: {}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))

#测试
#torch.save(Coder,'9.1.pkl')
#df=pd.DataFrame(Xs)
#df.to_csv('9.1.csv', index=False, header=False)
Coder = torch.load('CESDDM.pkl')
TP, TN, FP, FN = 0, 0, 0, 0
Coder.eval()
with torch.no_grad():
    for i in range(0, len3):
        gx = []
        max_gx = []
        for j in range(0, len2):
            gx.append(kernel(test_data[i], Xs[j], z))
        test_data1 = np.array(test_data[i])
        data_tensor = minmaxscaler(test_data1)
        data_tensor = torch.as_tensor(data_tensor)
        data_tensor = data_tensor.float().to(device)
        gx_tensor = np.array(gx)
        #max_gx = heapq.nlargest(10, gx_tensor)
        index = map(gx.index, heapq.nlargest(20, gx))
        #index = map(gx.index, heapq.nsmallest(20, heapq.nlargest(30, gx)))
        index = sorted(list(index))
        for k in range(len(index)):
            max_gx.append(gx[index[k]])
        loss1 = np.mean(max_gx)
        max_gx = torch.as_tensor(max_gx)
        max_gx = max_gx.float().to(device)
        #gx_tensor = torch.as_tensor(gx_tensor)
        #gx_tensor = gx_tensor.float().to(device)
        decoder = Coder(max_gx)
        loss = loss_func(decoder, data_tensor)
        b=test_label[i]
        if loss1 > 0.694 and loss < 0.0515:
            a = 0
        else:
            a = 1
        if a==1 and b==1:
            TP += 1
        if a==0 and b==0:
            TN += 1
        if a==1 and b==0:
            FP += 1
        if a==0 and b==1:
            FN += 1
P = TP / (TP + FP )
R = TP / (TP + FN )
F1=(2*P*R)/(P+R)
p1= (TP + TN)/ (TP+TN+FN+FP)
#acc=right/len3
print(P)
print(R)
print(p1)
print(F1)
print(TP)
print(TN)
print(FP)
print(FN)