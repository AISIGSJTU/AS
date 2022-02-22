import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear

from torch.autograd import Variable

__all__ = ['ResNet_vi', 'resnet20_vi', 'resnet32_vi', 'resnet44_vi', 'resnet56_vi', 'resnet110_vi', 'resnet1202_vi']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class RandBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, sigma_0, N, init_s, in_planes, planes, stride=1, option='A'):
        super(RandBasicBlock, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.conv1 = RandConv2d(self.sigma_0, self.N, self.init_s, in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = RandBatchNorm2d(self.sigma_0, self.N, self.init_s, planes)
        self.conv2 = RandConv2d(self.sigma_0, self.N, self.init_s, planes, planes, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.bn2 = RandBatchNorm2d(self.sigma_0, self.N, self.init_s, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                raise ValueError("Not implemented!")
                # self.shortcut = nn.Sequential(
                #     RandConv2d(self.sigma_0, self.N, self.init_s, in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                #     RandBatchNorm2d(self.sigma_0, self.N, self.init_s, self.expansion * planes)
                # )

    def forward(self, x):
        out, kl_sum, sample, fix = x
        out, kl = self.conv1(out, sample=sample, fix=fix)
        kl_sum += kl

        out, kl = self.bn1(out, sample=sample, fix=fix)
        kl_sum += kl

        out = F.relu(out)

        out, kl = self.conv2(out, sample=sample, fix=fix)
        kl_sum += kl

        out, kl = self.bn2(out, sample=sample, fix=fix)
        kl_sum += kl

        out += self.shortcut(x[0])
        out = F.relu(out)
        return out, kl_sum, sample, fix


class ResNet_vi(nn.Module):
    def __init__(self, sigma_0, N, init_s, block, num_blocks, num_classes=10):
        super(ResNet_vi, self).__init__()
        self.in_planes = 16
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.conv1 = RandConv2d(sigma_0, N, init_s, 3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = RandBatchNorm2d(sigma_0, N, init_s, 16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = RandLinear(sigma_0, N, init_s, 64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.sigma_0, self.N, self.init_s, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, penultimate=False, sample=True, fix=False):
        kl_sum = 0
        out, kl = self.conv1(x, sample=sample, fix=fix)
        kl_sum += kl

        out, kl = self.bn1(out, sample=sample, fix=fix)
        kl_sum += kl

        out = F.relu(out)

        out = self.layer1((out, kl_sum, sample, fix))
        out = self.layer2(out)
        out, kl_sum, _, _ = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if penultimate:
            return out
        out, kl = self.linear(out, sample=sample, fix=fix)
        kl_sum += kl
        return out, kl_sum


def resnet20_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202_vi(N, sigma_0=0.15, init_s=0.15, num_classes=10):
    return ResNet_vi(sigma_0, N, init_s, RandBasicBlock, [200, 200, 200], num_classes=num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet_vi.py'):
            print(net_name)
            test(globals()[net_name]())
            print()
