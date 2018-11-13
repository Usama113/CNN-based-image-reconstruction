import numpy as np
import _pickle as cPickle
from torch.utils import data
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init
from torch.autograd import Function
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from scipy.signal import hilbert

file='/data/CNN_partiel/pickle/train_x.pkl'
with open(file,'rb') as pickled_file:
    data1=cPickle.load(pickled_file)
print(data1.shape)
print(type(data1))
data1=data1.astype('float32')

file='/data/CNN_partiel/pickle/train_y.pkl'
with open(file,'rb') as pickled_file:
    target1=cPickle.load(pickled_file)
print(target1.shape)
print(type(target1))
target1=target1.astype('float32')


def check_dims(x1, x2):
    diffX = x1.data.shape[2] - x2.data.shape[2]
    diffY = x1.data.shape[3] - x2.data.shape[3]
    if (diffY == 1):
        x2 = F.pad(x2, (0, 1,
                        0, 0))
    if (diffX == 1):
        x2 = F.pad(x2, (0, 0,
                        0, 1))
    # print('####################')
    # print(x1.data.shape)
    # print(x2.data.shape)
    x = torch.cat([x2, x1], dim=1)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)





def preserve_dims(x1):
    x1 = F.pad(x1, (0, 1,
                    0, 0))
    # print(x1.data.shape)
    return x1


class Maxout(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        max_out = 4  # Maxout Parameter
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x = x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices = indices
        ctx.max_out = max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1, indices, max_out = ctx.saved_variables[0], Variable(ctx.indices), ctx.max_out
        input = input1.clone()
        for i in range(max_out):
            a0 = indices == i
            input[:, i:input.data.shape[1]:max_out] = a0.float() * grad_output

        return input


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.d1seq = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 256, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(256))
        self.mo1 = Maxout.apply
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=(2, 2))

        self.d2seq = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 512, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(512)
                                   )
        self.mo2 = Maxout.apply
        '''
        self.mp2=nn.MaxPool2d(kernel_size=2,stride=(2,2))


        self.d3seq=nn.Sequential(nn.Conv2d(128,256,kernel_size=(9,3),padding=(4,1)),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,kernel_size=(3,1),padding=(1,0)),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,1024,kernel_size=(3,1),padding=(1,0)),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())
        self.mo3=Maxout.apply
        self.mp3=nn.MaxPool2d(kernel_size=2,stride=(2,2))


        self.d4seq=nn.Sequential(nn.Conv2d(256,2048,kernel_size=(9,3),padding=(4,1)),
                                nn.BatchNorm2d(2048))
        self.mo4=Maxout.apply
        self.mp4=nn.MaxPool2d(kernel_size=2,stride=(2,1))


        self.d5seq=nn.Sequential(nn.Conv2d(512,1024,kernel_size=(3,1),padding=(1,0)),
                                nn.BatchNorm2d(1024),
                                nn.ReLU())


        self.up5=nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=2,stride=(2,1)),
                               nn.ReLU())
        self.u4seq=nn.Sequential(nn.Conv2d(1024,512,kernel_size=(3,1),padding=(1,0)),
                                  nn.BatchNorm2d(512),
                                  nn.ReLU())


        self.up4=nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=2,stride=(2,2)),
                                nn.ReLU())
        self.u3seq=nn.Sequential(nn.Conv2d(512,256,kernel_size=(9,3),padding=(4,1)),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU())


        self.up3=nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                              nn.ReLU(),
                              nn.Conv2d(256,128,kernel_size=(9,3),padding=(4,1)))
        self.u2seq=nn.Sequential(nn.Conv2d(256,128,kernel_size=(9,3),padding=(4,1)),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU())

        '''
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 64, kernel_size=(9, 3), padding=(4, 1)))
        self.u1seq = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(9, 3), padding=(4, 1)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 1, kernel_size=(1, 1)))

    def forward(self, x):
        down1 = self.mo1(self.d1seq(x))
        '''down2=self.d2seq(self.mp1(preserve_dims(down1)))
        down3=self.d3seq(self.mp2(preserve_dims(down2)))
        down4=self.d4seq(self.mp3(preserve_dims(down3)))
        down5=self.d5seq(self.mp4(preserve_dims(down4)))
        '''
        down2 = self.mo2(self.d2seq(self.mp1(down1)))
        # down3=self.mo3(self.d3seq(self.mp2(down2)))
        # down4=self.mo4(self.d4seq(self.mp3(down3)))
        # down5=self.d5seq(self.mp4(down4))

        '''print(down1.data.shape)
        print(down2.data.shape)
        print(down3.data.shape)
        print(down4.data.shape)
        print(down5.data.shape)
        '''
        # uc4=self.up5(down5)
        # print(uc4.data.shape)
        # up4=self.u4seq(check_dims(down4,uc4))
        # print(down3.data.shape)

        # uc3=self.up4(up4)

        # print(uc3.data.shape)
        # up3=self.u3seq(check_dims(down3,uc3))

        # uc2=self.up3(up3)
        # up2=self.u2seq(check_dims(down2,uc2))

        # uc1=self.up2(up2)
        # up1=self.u1seq(check_dims(down1,uc1))

        # uc3=self.up4(down4)
        # up3=self.u3seq(check_dims(down3,uc3))

        # uc2=self.up3(down3)
        # up2=self.u2seq(check_dims(down2,uc2))

        uc1 = self.up2(down2)
        up1 = self.u1seq(check_dims(down1, uc1))
        return up1


criterion=nn.MSELoss()
cnn=unet()
cnn.apply(weights_init)
cnn.cuda()
learning_rate=0.0001
optimizer=torch.optim.Adam(cnn.parameters(),lr=learning_rate)
#scheduler = MultiStepLR(optimizer, milestones=[300,4000], gamma=0.1)


def get_batch(data1, target1):
    n_examples = data1.shape[0] - 1
    i1 = random.randint(0, n_examples)
    i2 = random.randint(0, n_examples)
    i3 = random.randint(0, n_examples)
    i4 = random.randint(0, n_examples)
    i5 = random.randint(0, n_examples)
    '''i6=random.randint(0,n_examples)
    i7=random.randint(0,n_examples)
    i8=random.randint(0,n_examples)
    i9=random.randint(0,n_examples)
    i10=random.randint(0,n_examples)
    '''
    # d,t=torch.Tensor(data1[np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])]),torch.Tensor(target1[np.array([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])])
    d, t = torch.Tensor(data1[np.array([i1, i2, i3, i4, i5])]), torch.Tensor(target1[np.array([i1, i2, i3, i4, i5])])

    d = Variable(d).cuda()
    t = Variable(t).cuda()
    return d, t


epochs = 200
# f1=open('log_2000_l2_Maxout_pytorch.txt','w')
# f2=open('log_500_l2_theano_indices.txt','r')
results = {}
for epoch in range(epochs):
    print("Epoch:", epoch)

    d, t = get_batch(data1, target1)

    optimizer.zero_grad()
    results = cnn(d)

    loss = criterion(results, t)
    print('Loss:', loss.data[0])
    loss.backward()
    optimizer.step()
    # f1.write('Loss:'+str(loss.data[0])+'\n')



    print()
    print('############################')
    # f1.close()
    # epochs=7000
