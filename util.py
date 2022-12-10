import random
import torch
import numpy as np
import argparse
import pdb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def grad_param(net, mode=True):
    # freeze all layers but the last fc
    for name, param in net.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = mode


def count_param(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_train = sum([np.prod(p.size()) for p in model_parameters])
    param_all = sum([np.prod(p.size()) for p in model.parameters()])
    return param_all, param_train


def print_param(model):
    # print('\n--- frozen param ---\n')
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name)
    print('\n--- trainable param ---\n')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            if 'bias' in name:
                print(param.data[0])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
