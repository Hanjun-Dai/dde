from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
import random
from dde.common.cmd_args import cmd_args
from dde.unconditional import train_loop

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    data_file = '../../../data/synthetic/two_moon_train.npy'
    train_set = np.load(data_file)
    data_file = '../../../data/synthetic/two_moon_test.npy'
    test_set = np.load(data_file)
    train_loop(train_set, test_set)
