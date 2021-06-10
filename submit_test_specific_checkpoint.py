#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:21:59 2021

@author: ombretta
"""

import os 

submit_on_cluster = True

cluster_text = ''

if submit_on_cluster:
    cluster_text = '#!/bin/sh\n'+\
    '#SBATCH --partition=general\n'+\
    '#SBATCH --qos=short\n'+\
    '#SBATCH --time=4:00:00\n'+\
    '#SBATCH --ntasks=1\n'+\
    '#SBATCH --mail-type=END\n'+\
    '#SBATCH --cpus-per-task=2\n'+\
    '#SBATCH --mem=4000\n'+\
    '#SBATCH --gres=gpu:1\n'+\
    'module use /opt/insy/modulefiles\n'+\
    'module load cuda/10.0 cudnn/10.0-7.6.0.64\n'+\
    'srun ' #,jobspernode:one:1\n'+\
    
text = cluster_text + "python test_specific_checkpoint.py "

res_root = "results/" 
r = "movingmnist_longterm_vidbagnet_tem_9_256frames_32size_bs8_0" 
test_data_path = "../movingMNIST/movingmnistdata_longterm_permutedtestonly_permuted" 
annotation_path = "../movingMNIST/movingmnistdata_longterm_permutedtestonly_permuted/mnist_json_30.json"
checkpoint_epoch = 45

text += " --res_root="+res_root 
text += " --r="+r 
text += " --test_data_path="+test_data_path 
text += " --annotation_path="+annotation_path 
text += " --checkpoint_epoch="+checkpoint_epoch 

print(text)

if submit_on_cluster:
    with open("test_checkpoint.sbatch", "w") as file:
        file.write(text)
    os.system("sbatch test_checkpoint.sbatch")    
else:
    os.system(text)