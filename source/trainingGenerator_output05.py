#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:47:38 2018

@author: quanle
"""

from PIL import Image, ImageOps
from chainer import using_config
from chainer.cuda import to_cpu
from chainer.serializers import load_npz
from errno import EEXIST
from glob import glob
from os import makedirs
from os.path import split
from tqdm import tnrange
# from utils02 import Model

from PIL import Image, ImageOps
from chainer import ChainList
from chainer.dataset import DatasetMixin
import chainer.functions as F
import chainer.links as L
import numpy as np

class Dataset(DatasetMixin):
    def __init__(self, fp):
        super(Dataset, self).__init__()

        self.fp = fp

    def __len__(self):
        return len(self.fp[0])

    def get_example(self, i):
        #x = Image.open(self.fp[0][i]).convert('RGB').resize((138, 200), Image.LANCZOS)
        x = Image.open(self.fp[0][i]).convert('L').resize((138, 200), Image.LANCZOS)
        y = Image.open(self.fp[1][i]).convert('L').resize((138, 200), Image.LANCZOS)
# 
#         return np.asarray(x, 'f').transpose(2, 0, 1), np.asarray(y, 'f')[None]
        return np.asarray(x, 'f')[None], np.asarray(y, 'f')[None]

class ResidualBlock(ChainList):
    def __init__(self):
        super(ResidualBlock, self).__init__(
            L.Convolution2D(128, 128, 3, pad = 1),
            L.BatchNormalization(128),
            L.Convolution2D(128, 128, 3, pad = 1),
            L.BatchNormalization(128)
        )

    def __call__(self, x):
        return x + self[3](self[2](F.relu(self[1](self[0](x)))))

    
class ResidualBlock_64(ChainList):
    def __init__(self):
        super(ResidualBlock_64, self).__init__(
            L.Convolution2D(64, 64, 3, pad = 1),
            L.BatchNormalization(64),
            L.Convolution2D(64, 64, 3, pad = 1),
            L.BatchNormalization(64)
        )

    def __call__(self, x):
        return x + self[3](self[2](F.relu(self[1](self[0](x)))))
    
    
class Model(ChainList):
    def __init__(self):
        super(Model, self).__init__(
            L.Convolution2D(1, 32, 9, pad = 4, nobias = True),
            L.BatchNormalization(32),
            L.Convolution2D(32, 64, 3, 2, 1, True),
            L.BatchNormalization(64),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            ResidualBlock_64(),
            L.Deconvolution2D(64, 32, 3, 2, 1, True, (200, 138)),
            L.BatchNormalization(32),
            L.Convolution2D(32, 1, 9, pad = 4)
        )

    def __call__(self, x):
        for i in range(len(self)):
#             print x.shape
            x = F.relu(self[i](x)) if i in (1, 3, 5, 12, 14) else self[i](x)

        return 127.5 * F.tanh(x) + 127.5
  
device = 0 # -1 for CPU or GPU ID (0, 1, etc.) for GPU
fp = sorted(glob('../data_set/training_output_04/*.jpg'))
model = Model()

load_npz('../model_step05.npz', model)

if device >= 0:
    model.to_gpu(device)

try:
    makedirs('../data_set/training_output_05')
except OSError as exception:
    if exception.errno != EEXIST:
        raise

for i in xrange(len(fp)):
    print i
    x = model.xp.asarray(Image.open(fp[i]).convert('L').resize((138, 200), Image.LANCZOS), 'f')[None]
#     print x.shape

    with using_config('train', False):
        y_hat = Image.fromarray(to_cpu(model(x.reshape(1,1,200,138)).data[0, 0]).astype('uint8')).resize((275, 400), Image.LANCZOS)

    y_hat.save('../data_set/training_output_05/{}'.format(split(fp[i])[1])) # *contents* of the validation_output directory
                                                                           # should be zipped and submitted
