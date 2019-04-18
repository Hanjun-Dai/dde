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


class UniformProposal(nn.Module):
    def __init__(self, dataset):
        super(UniformProposal, self).__init__()
        self.X = torch.tensor(dataset, dtype=torch.float32)
        self.box_min, _ = torch.min(self.X, dim=0)
        self.box_max, _ = torch.max(self.X, dim=0)
        self.dist = torch.distributions.uniform.Uniform(self.box_min,
                                                        self.box_max)

    def forward(self, num_samples = cmd_args.batch_size):
        samples = self.dist.sample(torch.Size([num_samples]))
        if cmd_args.ctx == 'gpu':
            samples = samples.cuda()
        return samples

class NormalProposal(nn.Module):
    def __init__(self, sample_dim):
        super(NormalProposal, self).__init__()
        self.sample_dim = sample_dim

    def forward(self, num_samples = cmd_args.batch_size):
        z = torch.Tensor(num_samples, self.sample_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        z = Variable(z)
        return z
    
    def pdf(self, x):
        t = torch.exp(-x ** 2 / 2.0)
        return t / np.sqrt(2 * np.pi)

class GaussianProposal(nn.Module):
    def __init__(self, sample_dim, mu, sigma):
        super(GaussianProposal, self).__init__()
        self.sample_dim = sample_dim
        self.mu = Variable(torch.Tensor(np.copy(mu))).contiguous().view(1, -1)
        self.sigma = Variable(torch.Tensor(np.copy(sigma))).contiguous().view(1, -1)
        if cmd_args.ctx == 'gpu':
            self.mu = self.mu.cuda()
            self.sigma = self.sigma.cuda()

    def forward(self, num_samples = cmd_args.batch_size):
        z = torch.Tensor(num_samples, self.sample_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        z = Variable(z)
        z = z * self.sigma + self.mu
        return z

    def pdf(self, x):
        t = torch.exp(-(x - self.mu) ** 2 / 2.0 / self.sigma / self.sigma)
        t = t / np.sqrt(2 * np.pi) / self.sigma
        t = torch.prod(t, dim=-1, keepdim=True)
        return t
