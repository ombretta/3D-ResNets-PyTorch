#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:27:51 2020

@author: ombretta
"""

import os 

submit_on_cluster = True
pretrained = True

root_path = None
video_path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/instructional_videos/i3d_breakfast/data/processed/uniform_64_segments_raw.hdf5" # used as dataset_path
annotation_path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/instructional_videos/i3d_breakfast/data/processed/video_sets_split" # used as video_sets_split
result_path = "./kinetics_pretrained_breakfast_results/"
dataset = "breakfast"
n_classes = 10
sample_size = 224
sample_duration = 512
sample_t_stride = 1
no_hflip = True
no_mean_norm = True
no_std_norm = True
batch_size = 4
inference_batch_size = 4
n_val_samples = 1
n_epochs = 200
inference_no_average = True
n_threads = 1 # Number of threads for multi-thread loading (n_workers)
checkpoint = 1 # Trained model is saved at every this epochs
model = "resnet"
model_depth = 18 # Depth of resnet (10 | 18 | 34 | 50 | 101)
conv1_t_size = 7 # Kernel size in t dim of conv1
conv1_t_stride = 1 # Stride in t dim of conv1
no_max_pool = False # If true, the max pooling after conv1 is removed
resnet_shortcut = 'B' # Shortcut type of resnet (A | B)
resnet_widen_factor = 1.0 # The number of feature maps of resnet is multiplied by this value
wide_resnet_k = 2 # Wide resnet k
resnext_cardinality = 32 # ResNeXt cardinality
input_type = 'rgb'
manual_seed = 42
output_topk = 5 # Top-k scores are saved in json file
file_type = None # Normally 'jpg' or 'hdf5'
tensorboard = True
distributed = False # Use multi-processing distributed training to launch # Check !

text = ''

if submit_on_cluster:
    text += "#!/bin/sh\n" + \
    "#SBATCH --partition=general\n" + \
    "#SBATCH --qos=long\n" + \
    "#SBATCH --time=48:00:00\n" + \
    "#SBATCH --ntasks=1\n" + \
    "#SBATCH --mail-type=END\n" + \
    "#SBATCH --cpus-per-task=2\n" + \
    "#SBATCH --mem=16000\n" + \
    "#SBATCH --gres=gpu:4\n" + \
    "module use /opt/insy/modulefiles\n" + \
    "module load cuda/10.0 cudnn/10.0-7.6.0.64\n" + \
    "srun "
    
text += "python main.py "
if root_path: text += " --root_path=" + root_path
text += " --video_path=" + video_path
text += " --annotation_path=" + annotation_path
text += " --result_path=" + result_path
text += " --dataset=" + dataset
text += " --n_classes=" + str(n_classes)
text += " --sample_size=" + str(sample_size)
text += " --sample_duration=" + str(sample_duration)
text += " --sample_t_stride=" + str(sample_t_stride)
if no_hflip: text += " --no_hflip"
if no_mean_norm: text += " --no_mean_norm"
if no_std_norm: text += " --no_std_norm"
text += " --batch_size=" + str(batch_size)
text += " --inference_batch_size=" + str(inference_batch_size)
text += " --n_val_samples=" + str(n_val_samples)
text += " --n_epochs=" + str(n_epochs)
if inference_no_average: text += " --inference_no_average"
text += " --n_threads=" + str(n_threads)
text += " --checkpoint=" + str(checkpoint)
text += " --model=" + model
text += " --model_depth=" + str(model_depth)
text += " --conv1_t_size=" + str(conv1_t_size)
text += " --conv1_t_stride=" + str(conv1_t_stride)
if no_max_pool: text += " --no_max_pool"
text += " --resnet_shortcut=" + resnet_shortcut
text += " --resnet_widen_factor=" + str(resnet_widen_factor)
text += " --wide_resnet_k=" + str(wide_resnet_k)
text += " --resnext_cardinality=" + str(resnext_cardinality)
text += " --input_type=" + input_type
text += " --manual_seed=" + str(manual_seed)
text += " --output_topk=" + str(output_topk)
if file_type: text += " --file_type=" + file_type
if tensorboard: text += " --tensorboard"
if distributed: text += " --distributed"

if pretrained:
    text += " --pretrain_path=./pretrained_models/r3d18_K_200ep.pth"
    text += " --n_pretrain_classes=700"

print(text)

if submit_on_cluster:
    with open("breakfast_train.sbatch", "w") as file:
        file.write(text)
    os.system("sbatch breakfast_train.sbatch")    
else:
    os.system(text)
