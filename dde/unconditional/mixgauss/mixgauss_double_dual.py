from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from dde.common.cmd_args import cmd_args
from dde.unconditional import train_loop
import numpy as np
import torch
import random


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    data_file = '../../../data/synthetic/mixgauss_dim_%d_train.npy' % cmd_args.gauss_dim
    train_set = np.load(data_file)
    data_file = '../../../data/synthetic/mixgauss_dim_%d_test.npy' % cmd_args.gauss_dim
    test_set = np.load(data_file)
    train_loop(train_set, test_set)
