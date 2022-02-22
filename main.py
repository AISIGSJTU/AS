import os
import random
import functools
import shutil
import datetime

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import pickle

from utils import *
from model import *

parser = argparse.ArgumentParser(description='variance scaled weight perturbation')
parser.add_argument("-s", "--seed", type=int, default=42, help="The random seed")
parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="The learning rate")
parser.add_argument("-e", "--epoch", type=int, default=200, help="The total epoch")
parser.add_argument('--ratio', default=0.8, type=float, help='hyperparameter lambda for Adversarial Sampling')
parser.add_argument('--dataset', default='CIFAR-10', type=str, help='dataset')
parser.add_argument('--model', default='resnet20', type=str, help='network structure')
parser.add_argument('--samplings', default=1, type=int, help='sampling times')
parser.add_argument('--pretrain', default=False, type=bool, help='use pre training and Bayes finetune')
parser.add_argument('--notvi', help='not use BNN', dest="notvi", action='store_true')
args = parser.parse_args()
cudnn.benchmark = True

datapath = "~/data" 
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("Start time:", start_time.isoformat())
    seed = args.seed
    print("Use random seed ", seed)
    print("Use learning rate ", args.learning_rate)
    print("Use epoch", args.epoch)
    print("Use dataset", args.dataset)
    print("Sampling times", args.samplings)
    signature = "ratio"
    rootpath = f"results/{signature}"
    rootpath += f"_seed{seed}"
    rootpath += f'_l{args.learning_rate}'
    rootpath += f'_epoch{args.epoch}'
    rootpath += f'_ratio{args.ratio}'
    rootpath += f'_dataset{args.dataset}'
    rootpath += f'_model{args.model}'
    rootpath += f'_samplings{args.samplings}'
    rootpath += "_pretrain" if args.pretrain else ""
    rootpath += "_notvi" if args.notvi else ""
    rootpath += '/'
    if not os.path.isdir(rootpath):
        print("rootpath: ", rootpath)
        os.mkdir(rootpath)
    else:
        print("rootpath: ", rootpath)
        print("WARNING: rootpath exists")
    shutil.copyfile("main.py", rootpath + "main_bak.py")
    shutil.copyfile("utils.py", rootpath + "utils_bak.py")
    shutil.copyfile("model.py", rootpath + "model_bak.py")
    set_seed(seed)

    log_interval = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'CIFAR-10':
        trainset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR-100':
        trainset = datasets.CIFAR100(root=datapath, train=True, download=True,
                                     transform=transform_train)
        valset = datasets.CIFAR100(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError('Invalid dataset')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True,
                                              num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True,
                                            num_workers=0)

    if 'resnet' in args.model:
        net = ResNetWrapper(N=len(trainset), num_classes=num_classes, net=args.model, vi=not args.notvi)
    elif 'vgg' in args.model:
        net = VGG16VIWrapper(N=len(trainset), num_classes=num_classes, vi=not args.notvi)
    else:
        raise NotImplementedError('Invalid model')

    fim_traces = []
    train_losses = []
    train_precs = []
    test_losses = []
    test_precs = []

    lr = args.learning_rate

    if args.pretrain:
        pretrain_path = f"pretrain/{args.dataset}-{args.model}/{args.model}.pt"
        net_dict = net.model.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location='cuda:0')[
            'state_dict'].items()
        net_dict.update(
            {k.replace('weight', 'mu_weight').replace('bias', 'mu_bias'): v for
             k, v in pretrain_dict})
        net.model.load_state_dict(net_dict)
        print("Load from " + pretrain_path)
        net.validate(valloader, sample=False)

    for epoch in range(args.epoch):
        if epoch in [80, 140, 180]:
            lr /= 10

        loss, prec = net.fit(trainloader, lr=lr, epoch=epoch, optimizer="adam", ratio=args.ratio,
                             samplings=args.samplings)
        train_precs.append(prec)
        train_losses.append(loss)

        loss, prec = net.validate(valloader)
        test_precs.append(prec)
        test_losses.append(loss)
    print("Best prec, ", max(test_precs))
    net.save(rootpath + f"{args.model}.pt")
    end_time = datetime.datetime.now()
    print("End time:", end_time.isoformat())
    print("Total time:", end_time - start_time)
