from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dde.common.cmd_args import cmd_args
from dde.common.pytorch_initializer import weights_init

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    noise = sample_gumbel(logits.size())
    noise = noise.to(logits.device)
    y = logits + noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class MLPGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_out = None):
        super(MLPGenerator, self).__init__()
        self.is_flow = False
        self.act_out = act_out
        self.input_dim = input_dim
        self.output_dim = output_dim

        list_w = []
        list_w.append( nn.Linear(input_dim, hidden_dim) )
        for i in range(cmd_args.gen_depth - 2):
            list_w.append( nn.Linear(hidden_dim, hidden_dim) )

        list_w.append( nn.Linear(hidden_dim, output_dim) )

        self.list_w = nn.ModuleList(list_w)

        weights_init(self)

    def forward(self, num_samples = cmd_args.batch_size):
        z = torch.Tensor(num_samples, self.input_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        z = Variable(z)
        x = z
        for i in range(len(self.list_w)):
            h = self.list_w[i](x)
            if i + 1 < len(self.list_w):
                x = F.relu(h)
            else:
                x = h
        if self.act_out is not None:
            return self.act_out(x)
        return x


class CondMLPGenerator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, output_dim, act_out = None):
        super(CondMLPGenerator, self).__init__()
        self.is_flow = False
        self.act_out = act_out
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.input_dim = z_dim * 2
        self.output_dim = output_dim

        self.x2h = nn.Linear(x_dim, z_dim)
        list_w = []
        list_w.append( nn.Linear(self.input_dim, hidden_dim) )
        for i in range(cmd_args.gen_depth - 2):
            list_w.append( nn.Linear(hidden_dim, hidden_dim) )

        list_w.append( nn.Linear(hidden_dim, output_dim) )

        self.list_w = nn.ModuleList(list_w)

        weights_init(self)
    
    def forward(self, x):
        num_samples = x.shape[0]
        z = torch.Tensor(num_samples, self.z_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        z = Variable(z)
        h1 = self.x2h(x)
        x = torch.cat((h1, z), dim=-1)
        for i in range(len(self.list_w)):
            h = self.list_w[i](x)
            if i + 1 < len(self.list_w):
                x = F.relu(h)
            else:
                x = h
        if self.act_out is not None:
            return self.act_out(x)
        return x
