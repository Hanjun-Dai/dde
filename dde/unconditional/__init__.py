from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')
from dde.common.cmd_args import cmd_args
from dde.common.f_family import KernelExpFamily, MLPEnergy, get_kernel_mat
from dde.common.gen_family import MLPGenerator
from dde.common.proposal_dist import UniformProposal
from dde.common.maxF_minNu import max_f
import matplotlib.pyplot as plt
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
from itertools import chain

def MMD(f, y, x=None, gamma=None):
    if x is None:
        x = f.X
    if gamma is None:
        gamma = f.gamma
    kxx = get_kernel_mat(x, x, gamma)
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    kxx[idx, idx] = 0.0
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    kyy[idx, idx] = 0.0
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd.item()

import seaborn as sns
import pandas as pd

def heatmap(f):
    w = 100
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, w)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.flatten(), yy.flatten()]).transpose()

    x = torch.Tensor(coords).contiguous()
    scores = torch.exp(f(x)).data.cpu().numpy()
    a = scores.reshape((w, w))

    plt.imshow(a, cmap='hot')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig("heat.pdf" , bbox_inches='tight')
    plt.close()

def joint_plot(dataset, samples):
    plt.scatter(dataset[:, 0], dataset[:, 1], c='r', marker='x')
    plt.scatter(samples[:, 0], samples[:, 1], c='b', marker='.')
    plt.legend(['training data', 'DDE sampled'])
    plt.axis('equal')
    plt.savefig("joint.pdf", bbox_inches='tight')
    plt.close()

def train_loop(dataset, test_set):
    data_dim = dataset.shape[1]
    assert cmd_args.f_type == 'exp'
    f_func = KernelExpFamily(dataset, bandwidth=cmd_args.f_bd)
    generator = MLPGenerator(input_dim=cmd_args.z_dim, 
                             hidden_dim=cmd_args.nn_hidden_size,
                             output_dim=data_dim)
    proposal_dist = UniformProposal(dataset)
    assert cmd_args.v_type == 'mlp'
    kl_v = MLPEnergy(input_dim=data_dim, 
                    hidden_dim=cmd_args.nn_hidden_size, 
                    depth=cmd_args.f_depth, 
                    output_dim=1)
    if cmd_args.ctx == 'gpu':
        f_func = f_func.cuda()
        generator = generator.cuda()
        kl_v = kl_v.cuda()

    if cmd_args.init_model_dump is not None:
        f_func.load_state_dict(torch.load(cmd_args.init_model_dump + '.f', map_location='cpu'))
        generator.load_state_dict(torch.load(cmd_args.init_model_dump + '.q', map_location='cpu'))
        kl_v.load_state_dict(torch.load(cmd_args.init_model_dump + '.v', map_location='cpu'))

    if cmd_args.phase == 'test':
        assert cmd_args.init_model_dump is not None
        x = Variable(torch.Tensor(test_set))
        if cmd_args.ctx == 'gpu':
            x = x.cuda()
        samples = generator(num_samples=test_set.shape[0])
        mmd = MMD(f_func, samples, x)
        print('mmd: %.8f' % mmd)
        sys.exit()
    if cmd_args.phase == 'heat':
        assert cmd_args.init_model_dump is not None
        heatmap(f_func)
        sys.exit()
    if cmd_args.phase == 'joint':
        assert cmd_args.init_model_dump is not None
        samples = generator(num_samples=dataset.shape[0])
        samples = samples.data.cpu().numpy()
        joint_plot(dataset, samples)
        sys.exit()
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.axis('equal')
    plt.savefig("%s/dataset.pdf" % cmd_args.save_dir, bbox_inches='tight')
    plt.close()
    optF = optim.Adam(f_func.parameters() , lr=cmd_args.learning_rate)
    optG = optim.Adam(generator.parameters(), lr=cmd_args.learning_rate)
    opt_kl = optim.RMSprop(kl_v.parameters(), lr=cmd_args.learning_rate)

    iter_g = 5
    sample_idxes = list(range(dataset.shape[0]))
    pbar = tqdm(range(cmd_args.num_epochs), unit='epoch')
    f_log = open('%s/log.txt' % cmd_args.save_dir, 'w')
    f_log.write('gamma: %.8f\n' % f_func.gamma)
    for epoch in pbar:
        random.shuffle(sample_idxes)
        f_loss_list = []
        q_loss_list = []
        for pos in range(0, len(sample_idxes) // cmd_args.batch_size):
            selected_idx = sample_idxes[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
            x_input = Variable(torch.from_numpy(dataset[selected_idx, :]))
            if cmd_args.ctx == 'gpu':
                x_input = x_input.cuda()
            f_loss = max_f(x_input,
                           f_func,
                           generator,
                           optF)
            f_loss_list.append(f_loss.item())
            for i in range(iter_g):
                for j in range(cmd_args.v_iter):
                    opt_kl.zero_grad()
                    prior_samples = proposal_dist().detach()
                    fake_samples = generator().detach()
                    kl_lb = torch.mean( kl_v(fake_samples) ) - torch.mean( torch.exp( kl_v(prior_samples) ) )
                    v_loss = -kl_lb
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(kl_v.parameters(), max_norm=cmd_args.grad_clip)
                    opt_kl.step()
                optG.zero_grad()
                fake_samples = generator()
                q_loss = -torch.mean(f_func(fake_samples)) + cmd_args.kl_lambda * torch.mean( kl_v(fake_samples) )
                q_loss.backward()
                optG.step()
                q_loss_list.append(q_loss.item())
        pbar.set_description("f_loss: %.4f, q_loss: %.4f" % (np.mean(f_loss_list), np.mean(q_loss_list)))

        if epoch % 100 == 0:
            samples = generator(num_samples=dataset.shape[0])
            mmd = MMD(f_func, samples)
            msg = 'epoch: %d, mmd: %.8f' % (epoch, mmd)
            print(msg)
            f_log.write('%s\n' % msg)
            samples = samples.data.cpu().numpy()
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.axis('equal')
            plt.savefig("%s/samples-%d.pdf" % (cmd_args.save_dir, epoch), bbox_inches='tight')
            plt.close()
            torch.save(f_func.state_dict(), cmd_args.save_dir + '/epoch-%d.f' % epoch)
            torch.save(generator.state_dict(), cmd_args.save_dir + '/epoch-%d.q' % epoch)
            torch.save(kl_v.state_dict(), cmd_args.save_dir + '/epoch-%d.v' % epoch)
    f_log.close()
