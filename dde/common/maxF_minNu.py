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

def calc_gradient_penalty(netD, real_data, fake_data, posterior_sampler = None):
    if len(real_data.size()) == 2:
        alpha = torch.rand(real_data.size()[0], 1)
    elif len(real_data.size()) == 4:
        alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    else:
        print('unknown dimension of input')
        sys.exit()
    
    alpha = alpha.expand(real_data.size())
    if cmd_args.ctx == 'gpu':
        alpha = alpha.cuda()    

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cmd_args.ctx == 'gpu':
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if posterior_sampler is not None:
        posterior_z = Variable(posterior_sampler(interpolates, energy_func))
        disc_interpolates = netD(interpolates, posterior_z)
    else:
        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cmd_args.ctx == 'gpu' else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    if cmd_args.gnorm_type == 'lp1':
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * cmd_args.gnorm_lambda
    elif cmd_args.gnorm_type == 'norm2':
        gradient_penalty = gradients.norm(2, dim=1).mean() * cmd_args.gnorm_lambda
    else:
        raise NotImplementedError
        
    return gradient_penalty

def max_f(x_input, energy_func, generator, optimizerF):
    optimizerF.zero_grad()

    f_x = energy_func(x_input)
    loss_true = -F.torch.mean(f_x)

    sampled_x = generator(x_input.size()[0]).detach()
    f_sampled_x = energy_func(sampled_x)
    loss_fake = F.torch.mean(f_sampled_x)
    loss = loss_true + loss_fake

    if cmd_args.gnorm_lambda > 0:
        loss += calc_gradient_penalty(energy_func, x_input.data, sampled_x.data)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(energy_func.parameters(), max_norm=cmd_args.grad_clip)
    optimizerF.step()
    return loss

def min_nu(energy_func, flow, optimizerNu):
    optimizerNu.zero_grad()
    sampled_x, ll = flow()

    f_sampled_x = energy_func(sampled_x)
    loss = -F.torch.mean(f_sampled_x) + cmd_args.kl_lambda * F.torch.mean(ll)

    loss.backward()
    optimizerNu.step()
    return loss
