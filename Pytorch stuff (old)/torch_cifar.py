import os

from torchvision.datasets.cifar import CIFAR10
from fastai.conv_learner import *
from fastai.layers import Flatten

from torch import nn
import torch.nn.functional as F
import numpy as np

PATH = 'data'

def main():
    os.makedirs(PATH, exist_ok=True)
    trn_ds = CIFAR10(PATH, train=True, download=True)
    tst_ds = CIFAR10(PATH, train=False, download=True)
    trn = trn_ds.train_data.astype('float32')/255, np.array(trn_ds.train_labels)
    tst = tst_ds.test_data.astype('float32')/255, np.array(tst_ds.test_labels)

    sz, bs = 32, 128
    stats = (np.array([ 0.4914 ,  0.48216,  0.44653]),
             np.array([ 0.24703,  0.24349,  0.26159]))
    aug_tfms = [RandomFlip(), Cutout(1, 16)]
    tfms = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, pad=4)

    data = ImageClassifierData.from_arrays(PATH, trn, tst, bs=bs, tfms=tfms)
    wrn = WideResNet(n_grps=3, N=4, k=10)
    learn = ConvLearner.from_model_data(wrn, data)
    train(learn)


class BasicBlock(nn.Module):
    def __init__(self, inf, outf, stride, drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inf)
        self.conv1 = nn.Conv2d(inf, outf, kernel_size=3, padding=1, stride=stride, bias=False)
        self.drop = nn.Dropout(drop, inplace=True)
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(outf, outf, kernel_size=3, padding=1, stride=1, bias=False)
        if inf == outf:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inf), nn.ReLU(inplace=True),
                    nn.Conv2d(inf, outf, 3, padding=1, stride=stride, bias=False))

    def forward(self, x):
        x2 = self.conv1(F.relu(self.bn1(x)))
        x2 = self.drop(x2)
        x2 = self.conv2(F.relu(self.bn2(x2)))
        r = self.shortcut(x)
        return x2.add_(r)


class WideResNet(nn.Module):
    def __init__(self, n_grps, N, k=1, drop=0.3, first_width=16):
        super().__init__()
        # Double feature depth at each group,
        widths = [first_width]
        for grp in range(n_grps):
            widths.append(first_width*(2**grp)*k)
        layers = [nn.Conv2d(3, first_width, kernel_size=3, padding=1, bias=False)]
        for grp in range(n_grps):
            layers += self._make_group(N, widths[grp], widths[grp+1],
                                       (1 if grp == 0 else 2), drop)
        layers += [nn.BatchNorm2d(widths[-1]), nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1), Flatten(),
                   nn.Linear(widths[-1], 10)]
        self.features = nn.Sequential(*layers)

    def _make_group(self, N, inf, outf, stride, drop):
        group = list()
        for i in range(N):
            blk = BasicBlock(inf=(inf if i == 0 else outf), outf=outf,
                             stride=(stride if i == 0 else 1), drop=drop)
            group.append(blk)
        return group

    def forward(self, x):
        return self.features(x)


def train(learn):
    lr = 0.01
    wds = 5e-4
    for i, epochs in enumerate([60, 60, 40, 40]):
        learn.fit(lr, epochs, wds=wds, best_save_name=f'wrl-10-28-p{i}')
        lr /= 5


if __name__ == '__main__':
    main()
