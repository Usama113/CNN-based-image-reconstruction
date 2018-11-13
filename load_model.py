"""Utility functions."""
import operator
import datetime
import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert





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








class Maxout(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / 4)
        out_shape = (x.shape[0], feature_maps, 4, x.shape[2], x.shape[3])
        # print(out_shape)
        x= x.view(out_shape)
        # print(x.data.shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1,indices= ctx.saved_variables[0],Variable(ctx.indices)
        input = input1.clone()

        #print('indices:',indices.data.shape)
        #print('grad_output',grad_output.data.shape)
        #print('input',input.data.shape)

        a0 = indices == 0
        a1 = indices == 1
        a2 = indices == 2
        a3 = indices == 3
        input[:,0:input.data.shape[1]:4]=a0.float()*grad_output
        input[:, 1:input.data.shape[1]:4] = a1.float() * grad_output
        input[:, 2:input.data.shape[1]:4] = a2.float() * grad_output
        input[:, 3:input.data.shape[1]:4] = a3.float() * grad_output
        return input

















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









cnn=torch.load('model_l848.pth')



"""Contrast and resolution measures."""

#!/usr/bin/python3
import os
import h5py
import numpy as np
import operator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab

from scipy.signal import hilbert

from theano import config

################################################################################
# reading input
################################################################################
n_imgs = 80
#f = h5py.File('data/verasonics/data_170310_verasonics_test.h5', "r")
f = h5py.File('/data/CNN/us-beamformed/verasonics/data_170310_verasonics_test.h5', "r")
#f = h5py.File('/home/data/us-beamformed/verasonics/data_170310_verasonics_test.h5', "r")

x_mask = range(0, 128)
t_mask = range(0, 1332)  # rf_start -> 50 mm
z_mask = range(0, 1332)  # rf_start -> 50 mm

gamma = 0.3

c = f['/'].attrs['c']
rf_sampling_freq = f['/'].attrs['sampling_freq']

el_width = float(f['/'].attrs['el_width'])
el_kerf = float(f['/'].attrs['el_kerf'])

start_time = f['/'].attrs['start_time']

n_samples = int(f['/'].attrs['n_samples'])
n_elements = int(f['/'].attrs['n_elements'])
n_waves = int(f['/'].attrs['n_waves'])

dangle = float(f['/'].attrs['d_waves_angle']) * 360 / (2 * np.pi)

dx = el_width + el_kerf
dz = c / (2 * rf_sampling_freq)

zmap, xmap = np.meshgrid(
    dz * (np.arange(n_samples) + start_time * rf_sampling_freq),
    dx * (np.arange(n_elements) - (n_elements - 1) / 2),
    indexing='ij'
)

xmap = xmap[np.ix_(t_mask, x_mask)]
zmap = zmap[np.ix_(z_mask, x_mask)]

# all angles
a_mask = tuple(np.arange(n_waves))

angles = (np.arange(n_waves) - (n_waves - 1) / 2) * dangle

################################################################################
# coordinates definition
################################################################################
cr_coords = [
    {
        'img_i': 10,
        'center': (+.00013, .04428),
        'radius': .0037,
        'margin': .001
        # }, {
        #     'img_i': 20,
        #     'center': (-.00760, .04358),
        #     'radius': .0037,
        #     'margin': .001
        # }, {
        #     'img_i': 30,
        #     'center': (+.00904, .02400),
        #     'radius': .0018,
        #     'margin': .0005
        # }, {
        #     'img_i': 30,
        #     'center': (+.00933, .04362),
        #     'radius': .0018,
        #     'margin': .0005
    }
]

lr_coords = [
    {
        'img_i': 0,
        'center': (+.00058, .01293),
        'x_window': .001,
        'z_window': .001
    }, {
        'img_i': 0,
        'center': (+.00014, .02357),
        'x_window': .001,
        'z_window': .001
    }, {
        'img_i': 0,
        'center': (-.00016, .03322),
        'x_window': .001,
        'z_window': .001
#         }, {
#             'img_i': 10,
#             'center': (+.00043, .02335),
#             'x_window': .001,
#             'z_window': .001
#         }, {
#             'img_i': 20,
#             'center': (-.00701, .02293),
#             'x_window': .001,
#             'z_window': .001
#         }, {
#             'img_i': 20,
#             'center': (+.01355, .01279),
#             'x_window': .001,
#             'z_window': .001
#         }, {
#             'img_i': 20,
#             'center': (+.01296, .02330),
#             'x_window': .001,
#             'z_window': .001
#         }, {
#             'img_i': 20,
#             'center': (+.01296, .03303),
#             'x_window': .001,
#             'z_window': .001
    }
]

cr_img_idxs = [img['img_i'] for img in cr_coords]
lr_img_idxs = [img['img_i'] for img in lr_coords]
img_idxs = tuple(set(cr_img_idxs + lr_img_idxs))

rf_raw = np.asarray(f['rf'][img_idxs, ], dtype=config.floatX)
rf_raw = rf_raw[np.ix_(range(len(img_idxs)), a_mask, x_mask, t_mask)]
rf_raw = np.transpose(rf_raw, axes=(0, 1, 3, 2))

rf_fkmig = np.asarray(f['rf_fkmig'][img_idxs, ],
                      dtype=config.floatX)
rf_fkmig = rf_fkmig[np.ix_(range(len(img_idxs)), a_mask, x_mask, t_mask)]
rf_fkmig = np.transpose(rf_fkmig, axes=(0, 1, 3, 2))

rf_lumig = np.asarray(f['rf_lumig'][img_idxs, ],
                      dtype=config.floatX)
rf_lumig = rf_lumig[np.ix_(range(len(img_idxs)), a_mask, x_mask, t_mask)]
rf_lumig = np.transpose(rf_lumig, axes=(0, 1, 3, 2))

rf_target = np.mean(rf_fkmig, axis=1)


################################################################################
# stuff from utils.py
################################################################################
def cr(x, y):
    """Compute contrast ratio."""
    return 20 * np.log10(
        abs(np.mean(x) - np.mean(y)) /
        np.sqrt((np.var(x) + np.var(y)) / 2))

def envelope(x, axis=-2):
    """Envelope of RF signal x."""
    return abs(hilbert(x, axis=axis))

def normalize(x, axis=None, xmin=0.0, xmax=1.0):
    """Normalize x values."""
    assert (xmax > xmin), "Inconsistent xmin and xmax values (%f, %f)" % (
        xmin, xmax)

    if axis is None:
        axis = tuple(np.arange(x.ndim))
    else:
        axis = tuple(np.sort(axis))

    s = np.asarray(x.shape)
    s[np.asarray(axis)] = 1
    s = tuple(s)

    # between 0 and 1
    x = (x - np.min(x, axis=axis).reshape(s)) / \
        (np.max(x, axis=axis) - np.min(x, axis=axis)).reshape(s)
    # between min and max
    x = x * abs(xmax - xmin) + xmin
    return x

def resolution(x, y, th):
    """Resolution on x axis at y <= th."""
    ymax_i, ymax = max(enumerate(y), key=operator.itemgetter(1))
    th = -6

    return (
        dist_to_upper_threshold(
            x[ymax_i:], y[ymax_i:], th) +
        dist_to_upper_threshold(
            np.flip(x[:(ymax_i + 1)], 0), np.flip(y[:(ymax_i + 1)], 0), th))

def to_db(x):
    """Decibel scaling of X (positive signal)."""
    return 20 * np.log10(x / x.max())

def dist_to_upper_threshold(x, y, th):
    """X distance to Y below threshold."""
    d = 0
    xp = x[0]
    yp = y[0]

    for x, y in zip(x, y):
        if y <= th:
            d += abs(x - xp) * (th - yp) / (y - yp)
            break
        else:
            d += abs(x - xp)
        xp = x
        yp = y

    return d

################################################################################
# RF computing
################################################################################

rf = {}
temp_imgs=rf_fkmig[:,(0,15,30),:,:]

input_imgs=Variable(torch.Tensor(temp_imgs)).cuda()
results=cnn(input_imgs)


rf['model1']=np.reshape(results.data.cpu().numpy(),(results.data.shape[0],results.data.shape[2],results.data.shape[3]))

# recover RF from mean compounding
for k in np.arange(n_waves):
    # wide
    if k == 0:
        waves = [i for i, a in enumerate(angles) if a == 0]
    else:
        waves = np.round(np.linspace(0, n_waves - 1, k + 1)).astype(int)

    waves = tuple(waves)

#    print('comp-fk-wide-%i' % (k + 1))
    rf['comp-fk-wide-%i' % (k + 1)] = np.mean(
        rf_fkmig[:, waves, :, :], axis=1)
#    print('comp-lu-wide-%i' % (k + 1))
    rf['comp-lu-wide-%i' % (k + 1)] = np.mean(
        rf_lumig[:, waves, :, :], axis=1)

################################################################################
# CR and LR computation
################################################################################
#  [_cr[m] for m in l['names']], 'comp-lu-wide-%i' et cnn3 30° lu
_cr = {}
_lr = {}

for m in rf.keys():

    crs = []
    for c in cr_coords:
        i = img_idxs.index(c['img_i'])

        env = envelope(rf[m][i])
        bmode = np.power(normalize(env), gamma)

        r1 = c['radius'] - c['margin']
        r2 = c['radius'] + c['margin']
        r3 = np.sqrt(r1**2 + r2**2)

        cx = c['center'][0]
        cz = c['center'][1]

        dist = np.sqrt((xmap - cx)**2 + (zmap - cz)**2)

        m1 = dist <= r1
        m2 = np.all((dist > r2, dist <= r3), axis=0)

        crs.append(cr(bmode[m1], bmode[m2]))

    _cr[m] = np.mean(crs)

    lrs = []
    for l in lr_coords:
        i = img_idxs.index(l['img_i'])

        env = envelope(rf[m][i])

        cx = l['center'][0]
        cz = l['center'][1]

        # pixel centering
        cx = np.round((cx - np.min(xmap)) / dx) * dx + np.min(xmap)
        cz = np.round((cz - np.min(zmap)) / dz) * dz + np.min(zmap)

        res_x = np.linspace(-int(l['x_window'] / dx) * dx,
                            int(l['x_window'] / dx) * dx,
                            2 * int(l['x_window'] / dx) + 1)

        res_z = np.linspace(-int(l['z_window'] / dz) * dz,
                            int(l['z_window'] / dz) * dz,
                            2 * int(l['z_window'] / dz) + 1)

        res_xm = np.all((
            abs(zmap - cz) < (dz / 2),
            abs(xmap - cx) < l['x_window']), axis=0)

        res_zm = np.all((
            abs(zmap - cz) < l['z_window'],
            abs(xmap - cx) < (dx / 2)), axis=0)

        lrs.append(resolution(res_x, to_db(env[res_xm]), -6))

    _lr[m] = np.mean(lrs)

################################################################################
# usefull definitions
################################################################################

plt_series = {
#    'narrow': [
#        {
#            'label': 'standard',
#            'names': ['comp-lu-narrow-%i' % k for k in np.arange(n_waves) + 1],
#            'n_waves': np.arange(n_waves) + 1,
#            'linestyle': '-',
#            'marker': 'o',
#            'color': 'tab:blue'
#        }, {
#            'label': 'CNN',
#            # 'names': ('cnn1 lu', 'cnn2 2° lu', 'cnn3 2° lu'),
#            # 'n_waves': (1, 2, 3),
#            'names': ('cnn3 2° lu', ) * 31,
#            'n_waves': np.arange(n_waves) + 1,
#            'linestyle': '--',
#            'marker': '^',
#            'color': 'tab:orange'
#        }],
    'wide': [
        {
            'label': 'standard',
            'names': ['comp-lu-wide-%i' % k for k in np.arange(n_waves) + 1],
            'n_waves': np.arange(n_waves) + 1,
            'linestyle': '-',
            'marker': 's',
            'color': 'tab:blue'
        }, {
            'label': 'CNN',
            # 'names': ('cnn1 lu', 'cnn2 30° lu', 'cnn3 30° lu'),
            # 'n_waves': (1, 2, 3),
            'names': ('model1', ) * 31,
            'n_waves': np.arange(n_waves) + 1,
            'linestyle': '--',
            'marker': 'D',
            'color': 'tab:orange'
        }
    ]
}


################################################################################
# plotting figures
################################################################################
l_margin = 0.7
b_margin = 0.45
t_margin = 0.07
r_margin = 0.05

figsize = (6.4 * 0.25 + l_margin + r_margin,
           4.8 * 0.25 + b_margin + t_margin)
axrect = (l_margin / figsize[0],
          b_margin / figsize[1],
          1 - (l_margin + r_margin) / figsize[0],
          1 - (b_margin + t_margin) / figsize[1])


for s_name, s in plt_series.items():

    # Contrast ratio
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axrect)
    for l in s:
        # ax.fill_between(l['n_waves'],
        #                 [(_cr[m] + _cr_sd[m]) for m in l['names']],
        #                 [(_cr[m] - _cr_sd[m]) for m in l['names']],
        #                 facecolor=l['color'], alpha=0.3)
        ax.plot(l['n_waves'], [_cr[m] for m in l['names']],
                label=l['label'],
                # marker=l['marker'],
                linestyle=l['linestyle'],
                color=l['color'])
    ax.set_xlabel('number of PWs (standard)')
    # ax.set_ylabel('CR [dB]\nhigher is better')
    ax.set_ylabel('Contrast Ratio [dB]')
    ax.xaxis.set_ticks(np.arange(0, 31, 10))
    ax.yaxis.set_ticks([i for i in np.arange(0, 15, 1)
                        if i > ax.get_ylim()[0] and i < ax.get_ylim()[1]])
    ax.legend(loc='lower right')
    fig.savefig('meas_cr_{}.pdf'.format(s_name))

    # Lateral resolution
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axrect)
    for l in s:
        # ax.fill_between(l['n_waves'],
        #                 [(_lr[m] + _lr_sd[m]) * 1e3 for m in l['names']],
        #                 [(_lr[m] - _lr_sd[m]) * 1e3 for m in l['names']],
        #                 facecolor=l['color'], alpha=0.3)
        ax.plot(l['n_waves'], [_lr[m] * 1e3 for m in l['names']],
                label=l['label'],
                # marker=l['marker'],
                linestyle=l['linestyle'],
                color=l['color'])
    ax.set_xlabel('number of PWs (standard)')
    # ax.set_ylabel('LR [mm]\nlower is better')
    ax.set_ylabel('Lateral Resolution [mm]')
    ax.xaxis.set_ticks(np.arange(0, 31, 10))
    ax.yaxis.set_ticks([i for i in np.arange(0, 2, 0.2)
                        if i > ax.get_ylim()[0] and i < ax.get_ylim()[1]])
    ax.legend(loc='upper right')
    fig.savefig('meas_lr_{}.pdf'.format(s_name))

