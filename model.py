import numpy as np
import torch
import torch.nn as nn
import time
from torch.distributions import Categorical

from utils import *
from models.resnet_vi import resnet20_vi, resnet56_vi
from models.vgg_vi import VGG_vi
from models.vgg import VGG
from models.resnet import resnet20, resnet56


class NetWrapper():
    def __init__(self):
        cprint('c', '\nNet:')
        self.model = None

    def fit(self, train_loader, optimizer):
        raise NotImplementedError

    def predict(self, test_loader):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])


class ResNetWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, vi=True, num_classes=10, net='resnet20'):
        super(ResNetWrapper).__init__()
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if vi:
            if net == 'resnet20':
                self.model = resnet20_vi(N=N, num_classes=num_classes)
            elif net == 'resnet56':
                self.model = resnet56_vi(N=N, num_classes=num_classes)
        else:
            if net == 'resnet20':
                self.model = resnet20(num_classes=num_classes)
            elif net == 'resnet56':
                self.model = resnet56(num_classes=num_classes)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None, optimizer='adam', ratio=0.0,
            samplings=1):
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer {} not valid.".format(optimizer))
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double, vi=self.vi, ratio=ratio, samplings=samplings)

        return loss, prec

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi, sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He


class VGG16VIWrapper(NetWrapper):
    def __init__(self, N, half=False, cuda=True, double=False, num_classes=10, vi=True):
        super(VGG16VIWrapper).__init__()
        self.N = N
        self.vi = vi
        self.num_classes = num_classes
        if not vi:
            self.model = VGG(nclass=num_classes)
        else:
            self.model = VGG_vi(sigma_0=0.15, N=N, init_s=0.15, nclass=num_classes)
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None, optimizer='adam', ratio=0.0,
            samplings=1, sebr=0.0):
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer {} not valid.".format(optimizer))
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, optimizer, epoch, self.N, half=self.half, double=self.double, vi=self.vi, ratio=ratio, samplings=samplings)

        return loss, prec

    def validate(self, val_loader, sample=True):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, vi=self.vi, sample=sample)
        return loss, prec

    def sample_predict(self, x, Nsamples):
        self.model.eval()
        with torch.no_grad():
            predictions = x.data.new(Nsamples, x.shape[0], self.num_classes)

            Hs = []
            for i in range(Nsamples):
                y, kl = self.model(x)
                predictions[i] = y

                output = nn.functional.softmax(y)
                H = torch.distributions.Categorical(probs=output).entropy()
                Hs.append(H)

            Ha = sum(Hs) / Nsamples
            He = sum(torch.abs(Ha - i) for i in Hs) / Nsamples
        return predictions, Ha, He