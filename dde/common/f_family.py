from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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

class MLPEnergy(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, output_dim=1, act_out = None):
        super(MLPEnergy, self).__init__()
        self.act_out = act_out
        self.output_dim = output_dim
        self.depth = depth
        list_w = []
        if self.depth > 1:
            list_w.append( nn.Linear(input_dim, hidden_dim) )
            for i in range(self.depth - 2):
                list_w.append( nn.Linear(hidden_dim, hidden_dim) )

            list_w.append( nn.Linear(hidden_dim, self.output_dim) )
        else:
            list_w.append( nn.Linear(input_dim, self.output_dim) )

        self.list_w = nn.ModuleList(list_w)

        weights_init(self)

    def forward(self, x, mask = None):
        x = x.view(x.size()[0], -1)
        for i in range(self.depth):            
            if i + 1 == self.depth and cmd_args.fix_phi:
                x = x.detach()
            h = self.list_w[i](x)
            if i + 1 < self.depth:
                x = F.relu(h)
            else:
                x = h

        if self.act_out is not None:
            out = self.act_out(x)
        else:
            out = x
        if mask is not None:
            assert mask.size()[0] == out.size()[0] and mask.size()[1] == self.output_dim
            s = F.torch.sum(mask * out, dim=1, keepdim=True)
            return s
        return out

class CondMLPEnergy(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, depth, output_dim=1, act_out = None):
        super(CondMLPEnergy, self).__init__()
        self.act_out = act_out
        self.output_dim = output_dim
        self.depth = depth
        input_dim = 2 * x_dim
        self.y2h = nn.Linear(y_dim, x_dim)
        list_w = []
        if self.depth > 1:
            list_w.append( nn.Linear(input_dim, hidden_dim) )
            for i in range(self.depth - 2):
                list_w.append( nn.Linear(hidden_dim, hidden_dim) )

            list_w.append( nn.Linear(hidden_dim, self.output_dim) )
        else:
            list_w.append( nn.Linear(input_dim, self.output_dim) )

        self.list_w = nn.ModuleList(list_w)

        weights_init(self)

    def forward(self, x, y):
        x = x.view(x.size()[0], -1)
        hy = self.y2h(y)
        x = torch.cat([x, hy], dim=-1)
        for i in range(self.depth):            
            if i + 1 == self.depth and cmd_args.fix_phi:
                x = x.detach()
            h = self.list_w[i](x)
            if i + 1 < self.depth:
                x = F.relu(h)
            else:
                x = h

        if self.act_out is not None:
            out = self.act_out(x)
        else:
            out = x
        return out

def get_gamma(X, bandwidth):
    x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
    x_t = torch.transpose(X, 0, 1)
    x_norm_t = x_norm.view(1, -1)
    t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
    dist2 = F.relu(Variable(t)).detach().data

    d = dist2.cpu().numpy()
    d = d[np.isfinite(d)]
    d = d[d > 0]
    median_dist2 = float(np.median(d))
    print('median_dist2:', median_dist2)
    gamma = 0.5 / median_dist2 / bandwidth
    return gamma

def get_kernel_mat(x, landmarks, gamma):
    feat_dim = x.shape[1]
    batch_size = x.shape[0]
    x = x.view(1, -1).repeat(landmarks.shape[0], 1)  # db_size, bsize x feat_dim
    d = (x - landmarks.repeat(1, batch_size)) ** 2
    d_list = torch.split(d, feat_dim, dim=1)
    d = torch.sum(torch.cat(d_list, dim=0), dim=1)
    
    # get kernel matrix
    k = torch.exp(d * -gamma)
    k = k.view(batch_size, -1)
    return k

class CondKernelExpFamily(nn.Module):
    def __init__(self, X, Y, bd_x, bd_y):
        super(CondKernelExpFamily, self).__init__()
        self.X = torch.Tensor(X).contiguous()
        self.Y = torch.Tensor(Y).contiguous()
        self.basis_x = np.copy(X)
        self.basis_y = np.copy(Y)
        self.gamma_x = get_gamma(self.X, bd_x)
        self.gamma_y = get_gamma(self.Y, bd_y)
        print('gamma_x:', self.gamma_x)
        print('gamma_y:', self.gamma_y)
        if cmd_args.ctx == 'gpu':
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()
        self.X = Variable(self.X)
        self.Y = Variable(self.Y)
        self.db_size = self.X.shape[0]
        self.alpha = Parameter(torch.Tensor(self.db_size, 1).normal_(0, 0.1))

        self.kxx = get_kernel_mat(self.X, self.X, self.gamma_x)
        self.kyy = get_kernel_mat(self.Y, self.Y, self.gamma_y)
        self.kxxyy = self.kxx * self.kyy
        self.kxxyy = self.kxxyy.detach()
 
    def get_norm(self):
        if cmd_args.l2 <= 0:
            return 0.0
        at = torch.transpose(self.alpha, 0, 1)
        t = torch.matmul(at, self.kxxyy)
        norm2 = torch.matmul(t, self.alpha)
        return cmd_args.l2 * norm2

    def forward(self, x, y):
        kx = get_kernel_mat(x, self.X, self.gamma_x)
        ky = get_kernel_mat(y, self.Y, self.gamma_y)

        kxxyy = kx * ky
        scores = torch.matmul(kxxyy, self.alpha)
        return scores

class KernelExpFamily(nn.Module):        
    def __init__(self, dataset, bandwidth, bias=False):
        super(KernelExpFamily, self).__init__()
        self.has_bias = bias
        self.X = torch.tensor(dataset, dtype=torch.float32)
        assert bandwidth > 0
        self.gamma = get_gamma(self.X, bandwidth)
        self.bd = None
        print('gamma:', self.gamma)
        # else:
        #     self.bd = Parameter(torch.tensor(1.0 / np.abs(bandwidth), dtype=torch.float32))

        self.db_size = self.X.shape[0]
        self.alpha = Parameter(torch.Tensor(self.db_size, 1).normal_(0, 0.1))
        self.bias = Parameter(torch.Tensor(1).normal_(0, 0.1))
        if cmd_args.ctx == 'gpu':
            self.X = self.X.cuda() 

    def forward(self, x):
        if self.bd is None:
            gamma = self.gamma
        else:
            gamma = 0.5 / self.median_dist2 * self.bd
        k = get_kernel_mat(x, self.X, gamma)
        scores = torch.matmul(k, self.alpha)
        if self.has_bias:
            scores += self.bias
        return scores


def log_sum_exp(vec, dim, keepdim):
    vec = torch.sum(torch.exp(vec), dim=dim, keepdim=keepdim)
    return vec
