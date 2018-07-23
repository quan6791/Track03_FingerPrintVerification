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
        x = Image.open(self.fp[0][i]).convert('L').resize((138, 200), Image.LANCZOS)
        y = Image.open(self.fp[1][i]).convert('L').resize((138, 200), Image.LANCZOS)

        return np.asarray(x, 'f')[None], np.asarray(y, 'f')[None]

class ResidualBlock(ChainList):
    def __init__(self):
        super(ResidualBlock, self).__init__(
            L.Convolution2D(128, 128, 1, pad = 1),
            L.BatchNormalization(128),
            L.Convolution2D(128, 128, 1, pad = 1),
            L.BatchNormalization(128)
        )

    def __call__(self, x):
        return x + self[3](self[2](F.relu(self[1](self[0](x)))))

class Model(ChainList):
    def __init__(self):
        super(Model, self).__init__(
            L.Convolution2D(3, 32, 9, pad = 4, nobias = True),
            L.BatchNormalization(32),
            L.Convolution2D(32, 64, 3, 2, 1, True),
            L.BatchNormalization(64),
            L.Convolution2D(64, 128, 3, 2, 1, True),
            L.BatchNormalization(128),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            L.Deconvolution2D(128, 64, 3, 2, 1, True, (100, 69)),
            L.BatchNormalization(64),
            L.Deconvolution2D(64, 32, 3, 2, 1, True, (200, 138)),
            L.BatchNormalization(32),
            L.Convolution2D(32, 1, 9, pad = 4)
        )

    def __call__(self, x):
        for i in range(len(self)):
            x = F.relu(self[i](x)) if i in (1, 3, 5, 12, 14) else self[i](x)

        return 127.5 * F.tanh(x) + 127.5
