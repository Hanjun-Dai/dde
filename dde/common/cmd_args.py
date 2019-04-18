import argparse
from argparse import Namespace
import os
import pickle as cp
cmd_opt = argparse.ArgumentParser(description='Argparser for dde')
cmd_opt.add_argument('-saved_model', default=None, help='start from existing model')
cmd_opt.add_argument('-save_dir', default=None, help='save folder')
cmd_opt.add_argument('-cfg_file', default=None, help='cfg')
cmd_opt.add_argument('-init_model_dump', default=None, help='load model dump')

cmd_opt.add_argument('-ctx', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
cmd_opt.add_argument('-batch_size', type=int, default=100, help='minibatch size')
cmd_opt.add_argument('-iter_eval', type=int, default=100, help='iters per eval')

cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-gnorm_lambda', type=float, default=0.0, help='lambda for gradient norm')
cmd_opt.add_argument('-kl_lambda', type=float, default=1.0, help='lambda for kl')
cmd_opt.add_argument('-gnorm_type', type=str, default='lp1', help='type for gradient norm (lp1 || norm2)')
cmd_opt.add_argument('-f_depth', type=int, default=2, help='depth of f')
cmd_opt.add_argument('-num_epochs', type=int, default=50000, help='number of epochs')
cmd_opt.add_argument('-nn_hidden_size', type=int, default=128, help='dimension of mlp layers')
cmd_opt.add_argument('-learning_rate', type=float, default=0.001, help='init learning_rate')
cmd_opt.add_argument('-f_bd', type=float, default=0.01, help='kernel bandwidth of f')
cmd_opt.add_argument('-v_bd', type=float, default=0.01, help='kernel bandwidth of v')
cmd_opt.add_argument('-x_bd', type=float, default=0.01, help='kernel bandwidth of X')
cmd_opt.add_argument('-y_bd', type=float, default=0.01, help='kernel bandwidth of Y')
cmd_opt.add_argument('-l2', type=float, default=0.001, help='norm term of kernel exp')
cmd_opt.add_argument('-q0_std', type=float, default=1.0, help='std of q0')
cmd_opt.add_argument('-verb', type=int, default=1, help='display info')

cmd_opt.add_argument('-z_dim', type=int, default=64, help='dimension of latent variable')
cmd_opt.add_argument('-v_iter', type=int, default=5, help='iters of kl_dual update per each q update')
cmd_opt.add_argument('-grad_clip', type=int, default=5, help='clip of gradient')
cmd_opt.add_argument('-global_norm', type=int, default=0)

# args for gaussian experiment
cmd_opt.add_argument('-gauss_dim', type=int, default=2, help='dimension of gaussian')

# args for ring experiment
cmd_opt.add_argument('-ring_dim', type=int, default=2, help='dimension of ring data')
cmd_opt.add_argument('-fix_phi', type=int, default=0, help='fix phi or not')
cmd_opt.add_argument('-ring_radius', type=str, default='1,3,5', help='list of int, radius of each ring')
cmd_opt.add_argument('-f_type', type=str, default='exp', help='type for f (exp || mlp || ring || randfeat)')
cmd_opt.add_argument('-v_type', type=str, default='mlp', help='type for v (exp || mlp)')

cmd_opt.add_argument('-act', type=str, default='cos', help='type for activation (cos || relu)')
cmd_opt.add_argument('-num_rand_feats', type=int, default=1024, help='dimension of random features')

# args for generator
cmd_opt.add_argument('-flow_type', type=str, default='norm', help='type for flows (norm || iaf)')
cmd_opt.add_argument('-gen_depth', type=int, default=5, help='depth of flow')
cmd_opt.add_argument('-iaf_hidden', type=int, default=16, help='hidden dimension of autoregressive nn')

# args for r benchmark
cmd_opt.add_argument('-data_root', type=str, default=None, help='dataset root')
cmd_opt.add_argument('-data_name', type=str, default=None, help='dataset name')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
if cmd_args.cfg_file is not None:
    with open(cmd_args.cfg_file, 'rb') as f:
        d = cp.load(f)
    d = vars(d['args'])
    cur_args = vars(cmd_args)
    for key in d: 
        if key != 'save_dir' and key != 'seed' and key != 'verb':
            assert key in cur_args
            cur_args[key] = d[key]
        elif key == 'data_name':
            assert d[key] == cur_args[key]
    cmd_args = Namespace(**cur_args)
print(cmd_args)
