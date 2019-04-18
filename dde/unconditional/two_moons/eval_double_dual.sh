#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

epoch=11000
h=128
lambda=1.0
g_layers=4
z_dim=128
v_iter=3
clip=5
f_depth=3
lr=1e-4
f_bd=0.1
save_dir=$HOME/scratch/results/dde/two_moons/h-${h}-l-${lambda}-g-${g_layers}-z-${z_dim}-lr-${lr}-v-${v_iter}-c-${clip}-f-${f_depth}-fbd-${f_bd}

python two_moons_double_dual.py \
    -phase joint \
    -nn_hidden_size $h \
    -f_bd $f_bd \
    -f_depth $f_depth \
    -kl_lambda $lambda \
    -grad_clip $clip \
    -v_iter $v_iter \
    -z_dim $z_dim \
    -learning_rate $lr \
    -gen_depth $g_layers \
    -ctx cpu \
    -save_dir $save_dir \
    -init_model_dump epoch-$epoch
