#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:50:33 2018

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
from utils import Model
device = 0 # -1 for CPU or GPU ID (0, 1, etc.) for GPU
fp = sorted(glob('../data_set/validation_input/*.jpg'))
model = Model()

load_npz('../model.npz', model)

if device >= 0:
    model.to_gpu(device)
    
try:
    makedirs('../data_set/validation_output')
except OSError as exception:
    if exception.errno != EEXIST:
        raise

for i in xrange(len(fp)):
    print i
    x = model.xp.asarray(Image.open(fp[i]).convert('RGB').resize((138, 200), Image.LANCZOS), 'f').transpose(2, 0, 1)[None]

    with using_config('train', False):
        y_hat = Image.fromarray(to_cpu(model(x).data[0, 0]).astype('uint8')).resize((275, 400), Image.LANCZOS)

    y_hat.save('../data_set/validation_output/{}'.format(split(fp[i])[1])) # *contents* of the validation_output directory
                                                                           # should be zipped and submitted
                                                                           