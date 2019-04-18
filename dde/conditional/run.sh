#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_name=geyser
data_root=../../data/r_benchmark
dir=$(pwd)
cfg_file=$dir/default_configs/${data_name}-config.pkl
seed=1

save_dir=$HOME/scratch/results/dde/${data_name}-seed-${seed}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python r_double_dual.py \
    -seed $seed \
    -data_root $data_root \
    -cfg $cfg_file \
    -data_name $data_name \
    -save_dir $save_dir \
    -verb 1 \
