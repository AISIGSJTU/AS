import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def cprint(color: str, text: str, **kwargs) -> None:
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_beta(epoch_idx, N):
    return 1.0 / N / 100


def elbo(out, y, kl_sum, beta):
    ce_loss = F.cross_entropy(out, y)
    return ce_loss + beta * kl_sum



def train(train_loader, model, optimizer, epoch, N, half=True, print_freq=50, double=False, vi=True,
          ratio=0, samplings=1):
    """
        Run one train epoch
    """
    assert 0 <= ratio <= 1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    loss, prec1 = None, None
    for i, (input_data, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input_data.cuda()
        target_var = target
        if half:
            input_var = input_var.half()
        if double:
            input_var = input_var.double()

        # compute output
        if vi:
            if ratio > 1e-7:
                # Initialize epsilon with random unit Gaussian variable
                with torch.no_grad():
                    for module in model.modules():
                        if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                            module.eps_weight.normal_()
                            module.eps_weight.requires_grad = True
                            if module.eps_bias is not None:
                                module.eps_bias.normal_()
                                module.eps_bias.requires_grad = True

                alpha = 0.02
                iters = 5


                for index in range(iters):
                    for module in model.modules():
                        if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                            module.eps_weight.requires_grad = True
                            if module.eps_bias is not None:
                                module.eps_bias.requires_grad = True

                    routput, rkl = model(input_var, fix=True)
                    model.zero_grad()
                    rloss = - F.cross_entropy(routput, target) # Adversarial loss
                    rloss.backward()
                    # Updating
                    with torch.no_grad():
                        for module in model.modules():
                            if type(module).__name__ in ["RandConv2d", "RandLinear", "RandBatchNorm2d"]:
                                module.eps_weight -= alpha * module.eps_weight.grad.sign()
                                if module.eps_bias is not None:
                                    module.eps_bias -= alpha * module.eps_bias.grad.sign()
                routput, rkl = model(input_var, fix=True)
                rloss = F.cross_entropy(routput, target) # Adversarial loss
            else:
                rloss = 0

            if (1 - ratio) > 1e-7:
                # Prediction loss
                loss_sum = 0
                for s in range(samplings):
                    output, kl = model(input_var)
                    tloss = elbo(output, target, kl, get_beta(epoch, N))
                    loss_sum += tloss
                loss = loss_sum / samplings
            else:
                loss = 0
            # Total loss
            loss = (1 - ratio) * loss + ratio * rloss
            output, kl = model(input_var)
        else:
            output = model(input_var)
            loss = F.cross_entropy(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input_data.size(0))
        top1.update(prec1.item(), input_data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, half=True, print_freq=50, double=False, vi=True, fix=False, sample=True):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input_data.cuda()
            target_var = target.cuda()

            if half:
                input_var = input_var.half()
            if double:
                input_var = input_var.double()

            # compute output
            if vi:
                output, kl = model(input_var, fix=fix, sample=sample)
            else:
                output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return losses.avg, top1.avg


def data2img(data, path, reshape=True):
    if reshape:
        data = np.einsum('ijk->jki', data)
    data = (data * 255).round()
    data = np.uint8(data)
    img = Image.fromarray(data)
    img.save(path)



if __name__ == '__main__':
    for color in ['a', 'r', 'g', 'y', 'b', 'p', 'c', 'w']:
        cprint(color, color)
        cprint('*' + color, '*' + color)
