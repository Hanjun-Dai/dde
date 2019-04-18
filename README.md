# dde
Code for Kernel Exponential Family Estimation via Doubly Dual Embedding

## Requirement

The results reported in the paper are obtained with pytorch==0.3.1, but it also runs under pytorch 0.4.0 or newer versions, though some minor performance changes are observed. 

## Setup

At the root folder, do:

  `pip install -e .`
  
## Unconditional models

We train and evaluate on some synthetic datasets. To do so:

  `cd dde/unconditional/two_moons`
  
  `./run_double_dual.sh`

After training, run the evaluation script to get the visualization and quantitative results. 
  
  `./eval_double_dual.sh`
  
## Conditional models

We train and evaluate on 20 random splits of the benchmark datasets. 

To run the code, you need to provide the dataset name and random seed. Different random seeds correspond to different data splits. 

  `cd dde/conditiona`

  `./run.sh`
  
By default the script loads the default configurations of hyperparameters inside default_configs/ folder. You can also tune it a bit for your own dataset. The most important parameters are kernel bandwidths. 

