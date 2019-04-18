from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


from dde.common.cmd_args import cmd_args
from dde.common.f_family import CondKernelExpFamily, CondMLPEnergy
from dde.common.gen_family import CondMLPGenerator
from dde.common.proposal_dist import NormalProposal, GaussianProposal
from dde.common.pytorch_initializer import to_scalar
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
import pickle as cp
from dde.conditional.r_data_loader import load_dataset

def max_f(x_input, y_input, energy_func, generator, optimizerF):
    optimizerF.zero_grad()

    f_xy = energy_func(x_input, y_input)
    loss_true = -F.torch.mean(f_xy)

    sampled_y = generator(x_input).detach()
    f_sampled_y = energy_func(x_input, sampled_y)
    loss_fake = F.torch.mean(f_sampled_y)
    loss = loss_true + loss_fake
    if cmd_args.l2 >= 0:
        norm = energy_func.get_norm()
        loss = loss + norm
    loss.backward()
    optimizerF.step()
    return loss

def data_generator(x_train, y_train):
    sample_idxes = list(range(x_train.shape[0]))
    while True:
        random.shuffle(sample_idxes)
        for pos in range(0, len(sample_idxes) // cmd_args.batch_size):
            selected_idx = sample_idxes[pos * cmd_args.batch_size : (pos + 1) * cmd_args.batch_size]
            bsize = len(selected_idx)
            x_input = Variable(torch.from_numpy(x_train[selected_idx, :])).view(bsize, -1)
            y_input = Variable(torch.from_numpy(y_train[selected_idx, :])).view(bsize, -1)
            if cmd_args.ctx == 'gpu':
                x_input = x_input.cuda()
                y_input = y_input.cuda()
            yield x_input, y_input

def eval_obj(f_func, generator, kl_v, proposal_dist, x, y):
    bsize = x.shape[0]
    x = Variable(torch.from_numpy(x)).view(bsize, -1).contiguous()
    y = Variable(torch.from_numpy(y)).view(bsize, -1).contiguous()

    f_xy = f_func(x, y)
    samples = 10
    s = 0.0
    for _ in range(samples):
        sampled_y = generator(x).detach()
        f_sampled_y = f_func(x, sampled_y)
        s = s - torch.mean(f_sampled_y)
        
        prior_samples = proposal_dist(bsize).detach()
        fake_samples = generator(x).detach()
        kl_lb = torch.mean( kl_v(x, fake_samples) ) \
                            - torch.mean( torch.exp( kl_v(x, prior_samples) ) ) + 1
        # s = s + kl_lb
    s = s / samples
    s = s + torch.mean(f_xy) 
    return s.data[0]

def importance_sampling(f_func, x, y, dist_g):
    bsize = x.shape[0]
    x = Variable(torch.Tensor(x))
    x = x.contiguous().view(bsize, -1)
    y = Variable(torch.Tensor(y)).contiguous().view(bsize, -1)
    steps = 1000
    s = 0.0
    for t in range(steps):
        sampled_y = dist_g(bsize).detach()
        fy = torch.exp( f_func(x, sampled_y) ) * proposal_dist.pdf(sampled_y)
        gy = dist_g.pdf(sampled_y)
        s = s + fy / gy / steps
    log_f = f_func(x, y) + torch.log(proposal_dist.pdf(y) + 1e-32)
    ll = log_f - torch.log(s)
    t = proposal_dist.pdf(y)
    ll = torch.mean(ll)
    return to_scalar(ll)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    x_train, y_train, x_test, y_test = load_dataset()
    print('train', x_train.shape, y_train.shape)
    print('test', x_test.shape, y_test.shape)
    if cmd_args.batch_size > x_train.shape[0]:
        cmd_args.batch_size = x_train.shape[0]
    d = {'args': cmd_args}
    with open('%s/config.pkl' % cmd_args.save_dir, 'wb') as f:
        cp.dump(d, f, cp.HIGHEST_PROTOCOL)
    f_func = CondKernelExpFamily(x_train, y_train, bd_x=cmd_args.x_bd, bd_y=cmd_args.y_bd)
    generator = CondMLPGenerator(x_dim=x_train.shape[1],
                                 z_dim=cmd_args.z_dim,
                                 hidden_dim=cmd_args.nn_hidden_size,
                                 output_dim=y_train.shape[1])
    if cmd_args.v_type == 'mlp':
        kl_v = CondMLPEnergy(x_dim=x_train.shape[1],
                             y_dim=y_train.shape[1],
                             hidden_dim=cmd_args.nn_hidden_size, 
                             depth=cmd_args.f_depth, 
                             output_dim=1)
    if cmd_args.global_norm:
        y_all = np.concatenate([y_train, y_test], axis=0)
        proposal_dist = GaussianProposal(y_train.shape[1],
                                        np.mean(y_all, axis=0),
                                        np.std(y_all, axis=0) * cmd_args.q0_std)
        test_dist_g = proposal_dist
    else:
        proposal_dist = GaussianProposal(y_train.shape[1],
                                        np.mean(y_train, axis=0),
                                        np.std(y_train, axis=0) * cmd_args.q0_std)
        test_dist_g = GaussianProposal(y_train.shape[1],
                                    np.mean(y_test, axis=0),
                                    np.std(y_test, axis=0))
    if cmd_args.ctx == 'gpu':
        f_func = f_func.cuda()
        generator = generator.cuda()
        kl_v = kl_v.cuda()
    optF = optim.Adam(f_func.parameters() , lr=cmd_args.learning_rate)
    optG = optim.Adam(generator.parameters(), lr=cmd_args.learning_rate)
    opt_kl = optim.RMSprop(kl_v.parameters(), lr=cmd_args.learning_rate)
    iter_g = 5
    datagen = data_generator(x_train, y_train)
    if cmd_args.verb:
        pbar = tqdm(range(cmd_args.num_epochs), unit='epoch')
    else:
        pbar = range(cmd_args.num_epochs)
    f_log = open('%s/log.txt' % cmd_args.save_dir, 'w', buffering=1)
    for epoch in pbar:
        f_loss_list = []
        q_loss_list = []
        for pos in range(0, x_train.shape[0] // cmd_args.batch_size):
            x_input, y_input = next(datagen)
            f_loss = max_f(x_input,
                           y_input,
                           f_func,
                           generator,
                           optF)
            f_loss_list.append(to_scalar(f_loss))
            for i in range(iter_g):
                for j in range(cmd_args.v_iter):
                    x_input, _ = next(datagen)
                    opt_kl.zero_grad()
                    prior_samples = proposal_dist(x_input.shape[0]).detach()
                    fake_samples = generator(x_input).detach()
                    kl_lb = torch.mean( kl_v(x_input, fake_samples) ) \
                            - torch.mean( torch.exp( kl_v(x_input, prior_samples) ) )
                    v_loss = -kl_lb
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm(kl_v.parameters(), max_norm=cmd_args.grad_clip)
                    opt_kl.step()
                optG.zero_grad()
                x_input, _ = next(datagen)
                fake_samples = generator(x_input)
                q_loss = -torch.mean(f_func(x_input, fake_samples)) \
                        + cmd_args.kl_lambda * torch.mean( kl_v(x_input, fake_samples) )
                q_loss.backward()
                optG.step()
                q_loss_list.append(to_scalar(q_loss))
        if cmd_args.verb:
            pbar.set_description("f_loss: %.4f, q_loss: %.4f" % (np.mean(f_loss_list), np.mean(q_loss_list)))
        if epoch % cmd_args.iter_eval == 0:
            test_ll = importance_sampling(f_func, x_test, y_test, test_dist_g)
            train_ll = importance_sampling(f_func, x_train, y_train, proposal_dist)

            msg = 'epoch: %d, train: %.4f, test: %.4f' % (epoch, -train_ll, -test_ll)
            if cmd_args.verb:
                print(msg)
            f_log.write(msg + '\n')
    f_log.close()
