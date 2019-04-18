from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from dde.common.cmd_args import cmd_args
import sys
import csv
import numpy as np
import random


def normalize(d_train, d_test):
    m = np.mean(d_train, axis=0)
    s = np.std(d_train, axis=0) + 1e-16

    d_train = (d_train - m) / s
    d_test = (d_test - m) / s
    return d_train, d_test


def load_dataset():
    f_data = '%s/%s.npy' % (cmd_args.data_root, cmd_args.data_name)
    data = np.load(f_data)
    np.random.shuffle(data)    
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    num_samples = X.shape[0]
    X_train = X[:num_samples//2, :]
    Y_train = Y[:num_samples//2, :]

    X_test = X[num_samples//2:, :]
    Y_test = Y[num_samples//2:, :]    
    X_train, X_test = normalize(X_train, X_test)
    Y_train, Y_test = normalize(Y_train, Y_test)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    return X_train, Y_train, X_test, Y_test
