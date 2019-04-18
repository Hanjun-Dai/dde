from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from dde.common.cmd_args import cmd_args
import numpy as np
import argparse

true_mean = 1.0
true_std = 0.1

def get_diag_gaussian(num_samples=10000):
    return (np.random.randn(num_samples, cmd_args.gauss_dim) * true_std + true_mean).astype(np.float32)

def get_ring(num_samples=10000, sigma=0.1):
    D = cmd_args.ring_dim
    N = num_samples
    radia = np.array([int(x) for x in cmd_args.ring_radius.split(',')])
    assert D >= 2
    
    angles = np.random.rand(N) * 2 * np.pi
    noise = np.random.randn(N) * sigma
    
    weights = 2 * np.pi * radia
    weights /= np.sum(weights)
    
    radia_inds = np.random.choice(len(radia), N, p=weights)
    radius_samples = radia[radia_inds] + noise
    
    xs = (radius_samples) * np.sin(angles)
    ys = (radius_samples) * np.cos(angles)
    X = np.vstack((xs, ys)).T.reshape(N, 2)
    
    result = np.zeros((N, D))
    result[:, :2] = X
    if D > 2:
        result[:, 2:] = np.random.randn(N, D - 2) * sigma
    return result.astype(np.float32)