#imports

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

#loading the data
file='/data/CNN/pickle1/train_x1.pkl'
with open(file,'rb') as pickled_file:
    data1=cPickle.load(pickled_file)
print(data1.shape)
print(type(data1))
data1=data1.astype('float32')

file='/data/CNN/pickle1/train_y1.pkl'
with open(file,'rb') as pickled_file:
    target1=cPickle.load(pickled_file)
print(target1.shape)
print(type(target1))
target1=target1.astype('float32')

#weights initialization to xavier distribution
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


#The maxout module
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


#The CNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 256, kernel_size=(9, 3), padding=(4, 1))
        self.mo1=Maxout.apply
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(17, 5), padding=(8, 2))
        self.mo2 = Maxout.apply
        self.layer3 = nn.Conv2d(32, 64, kernel_size=(33, 9), padding=(16, 4))
        self.mo3 = Maxout.apply
        self.layer4 = nn.Conv2d(16, 32, kernel_size=(65, 17), padding=(32, 8))
        self.mo4 = Maxout.apply
        self.layer_out = nn.Conv2d(8, 4, kernel_size=(1, 1))
        self.mo5 = Maxout.apply

    def forward(self, x):
        out = self.mo1(self.layer1(x))
        out = self.mo2(self.layer2(out))
        out = self.mo3(self.layer3(out))
        out = self.mo4(self.layer4(out))
        out = self.mo5(self.layer_out(out))
        return out



#Batch Creation

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


#Setting up CNN

criterion=nn.MSELoss()
cnn=CNN()
cnn.apply(weights_init)
cnn.cuda()
learning_rate=0.0001
optimizer=torch.optim.Adam(cnn.parameters(),lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[300,4000], gamma=0.1)


#Training starts here

epochs = 2500
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

#Save the model

torch.save(cnn, './model_x1_epochs.pth')
torch.save(optimizer.state_dict(), './optimizer_x1.pth')
