#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:23:32 2020

@author: ombretta
"""

'''Arguments IN 3D ResNets PyTorch'''

import os 

submit_on_cluster = True
pretrained = False

text = ''

if submit_on_cluster:
    text = '#!/bin/sh\n'+\
    '#SBATCH --partition=general\n'+\
    '#SBATCH --qos=long\n'+\
    '#SBATCH --time=60:00:00\n'+\
    '#SBATCH --ntasks=1\n'+\
    '#SBATCH --mail-type=END\n'+\
    '#SBATCH --cpus-per-task=2\n'+\
    '#SBATCH --mem=16000\n'+\
    '#SBATCH --gres=gpu:4\n'+\
    'module use /opt/insy/modulefiles\n'+\
    'module load cuda/10.0 cudnn/10.0-7.6.0.64\n'+\
    'srun '
    
'''List of arguments from the repository:'''
# Root directory path
root_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/'
# Directory path of videos
video_path = 'UCF-101/jpg'
# Annotation file path
annotation_path = 'UCF-101/json_annotations/ucf101_01.json'
# Used dataset (activitynet | kinetics | ucf101 | hmdb51)
dataset = 'ucf101'
# Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)
n_classes = 101
# Number of classes of pretraining task. When using --pretrain_path, this must be set.
n_pretrain_classes = 700
# Pretrained model path (.pth).
pretrain_path = "3D-ResNets-PyTorch/pretrained_models/r3d50_K_200ep.pth"
# Module name of beginning of fine-tuning (conv1, layer1, fc, denseblock1, classifier, ...). The default means all layers are fine-tuned.
ft_begin_module = ''
# Height and width of inputs
sample_size = 64
# Temporal duration of inputs
sample_duration = 48
# If larger than 1, input frames are subsampled with the stride.
sample_t_stride = 1
# Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center. random | corner | center)
train_crop = 'random'
# Min scale for random cropping in training
train_crop_min_scale = 0.25
# Min scale for random cropping in training
train_crop_min_ratio = 0.75
# If true holizontal flipping is not performed.
no_hflip = False
# If true colorjitter is performed.
colorjitter = False
# Temporal cropping method in training. random is uniform. (random | center)
train_t_crop = 'random'
# Initial learning rate (divided by 10 while training by lr scheduler)
learning_rate = 0.01
# Momentum
momentum = 0.9
# dampening of SGD
dampening = 0.0
# Weight Decay
weight_decay = 1e-3
# dataset for mean values of mean subtraction (activitynet | kinetics | 0.5)
mean_dataset = 'kinetics'
# If true, inputs are not normalized by mean.
no_mean_norm = False
# If true, inputs are not normalized by standard deviation.
no_std_norm = False
# If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].
value_scale = 1
# Nesterov momentum
nesterov = False
# Currently only support SGD
optimizer = 'sgd'
# Type of LR scheduler (multistep | plateau)
lr_scheduler = 'multistep'
# Milestones of LR scheduler. See documentation of MultistepLR.
multistep_milestones = 30
# If true, overwriting multistep_milestones when resuming training.
overwrite_milestones = False
# Patience of LR scheduler. See documentation of ReduceLROnPlateau.
plateau_patience = 10
# Batch Size
batch_size = 16
# Batch Size for inference. 0 means this is the same as batch_size.
inference_batch_size = 0
# If true, SyncBatchNorm is used instead of BatchNorm.
batchnorm_sync = False
# Number of total epochs to run
n_epochs = 80
# Number of validation samples for each activity
n_val_samples = 3
# Save data (.pth) of previous training
resume_path = None
# If true, training is not performed.
no_train = False
# If true, validation is not performed.
no_val = False
# If true, inference is performed.
inference = False
# Used subset in inference (train | val | test)
inference_subset = 'val'
# Stride of sliding window in inference.
inference_stride = 16
# Cropping method in inference. (center | nocrop). When nocrop, fully convolutional inference is performed, and mini-batch consists of clips of one video.
inference_crop = 'center'
# If true, outputs for segments in a video are not averaged.
inference_no_average = False
# If true, cuda is not used.
no_cuda = False
# Number of threads for multi-thread loading
n_threads = 4
# Trained model is saved at every this epochs.
checkpoint = 5
# (resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | vidbagnet |
model = 'resnet' #'vidbagnet'
# Depth of resnet (10 | 18 | 34 | 50 | 101)
model_depth = 18
# Depth of resnet (9 | 17 | 33)
receptive_size = 33
# Kernel size in t dim of conv1.
conv1_t_size = 7
# Stride in t dim of conv1.
conv1_t_stride = 1
# If true, the max pooling after conv1 is removed. 
no_max_pool = False
# Shortcut type of resnet (A | B)
resnet_shortcut = 'B'
# The number of feature maps of resnet is multiplied by this value
resnet_widen_factor = 1.0
# Wide resnet k
wide_resnet_k = 2
# ResNeXt cardinality
resnext_cardinality = 32
# (rgb | flow)
input_type = 'rgb'
# Manually set random seed
manual_seed = 1
# If true, accimage is used to load images.
accimage = False
# Top-k scores are saved in json file.
output_topk = 5
# (jpg | hdf5)
file_type = 'jpg'
# If true, output tensorboard log file.
tensorboard = True
# Use multi-processing distributed training to launch N processes per node, which has N GPUs.
distributed = False
# url used to set up distributed training
dist_url = 'tcp://127.0.0.1:23456'
# number of nodes for distributed training
world_size = 1

text += "python main.py "

if root_path: text += " --root_path=" + root_path
text += " --video_path=" + video_path
text += " --annotation_path=" + annotation_path
text += " --dataset=" + dataset
text += " --n_classes=" + str(n_classes)
text += " --sample_size=" + str(sample_size)
text += " --sample_duration=" + str(sample_duration)
text += " --sample_t_stride=" + str(sample_t_stride)
text += " --train_crop=" + train_crop
text += " --train_crop_min_scale=" + str(train_crop_min_scale)
text += " --train_crop_min_ratio=" + str(train_crop_min_ratio)
if colorjitter: text += " --colorjitter"
text += " --train_t_crop=" + str(train_t_crop)
text += " --learning_rate=" + str(learning_rate)
text += " --momentum=" + str(momentum)
text += " --dampening=" + str(dampening)
text += " --weight_decay=" + str(weight_decay)
text += " --mean_dataset=" + mean_dataset
if no_mean_norm: text += " --no_mean_norm"
if no_std_norm: text += " --no_std_norm"
text += " --value_scale=" + str(value_scale)
if nesterov: text += " --nesterov"
text += " --optimizer=" + optimizer
text += " --lr_scheduler=" + lr_scheduler
text += " --multistep_milestones=" + str(multistep_milestones)
if overwrite_milestones: text += " --overwrite_milestones"
text += " --plateau_patience=" + str(plateau_patience)
text += " --inference_batch_size=" + str(inference_batch_size)
if batchnorm_sync: text += " --batchnorm_sync"
if resume_path: text += " --resume_path"
if no_train: text += " --no_train"
if no_val: text += " --no_val"
if inference: text += " --inference"
text += " --inference_subset=" + inference_subset
text += " --inference_stride=" + str(inference_stride)
text += " --inference_crop=" + inference_crop
if no_cuda: text += " --no_cuda"
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
text += " --receptive_size=" + str(receptive_size)
text += " --conv1_t_size=" + str(conv1_t_size)
text += " --conv1_t_stride=" + str(conv1_t_stride)
if no_max_pool: text += " --no_max_pool"
text += " --resnet_shortcut=" + resnet_shortcut
text += " --resnet_widen_factor=" + str(resnet_widen_factor)
text += " --wide_resnet_k=" + str(wide_resnet_k)
text += " --resnext_cardinality=" + str(resnext_cardinality)
text += " --input_type=" + input_type
text += " --manual_seed=" + str(manual_seed)
if accimage: text += " --accimage"
text += " --output_topk=" + str(output_topk)
if file_type: text += " --file_type=" + file_type
if tensorboard: text += " --tensorboard"
if distributed: text += " --distributed"
text += " --ft_begin_module="+ft_begin_module

#last
# Result directory path
results_root = '3D-ResNets-PyTorch/results/'
result_path = dataset + "_" + model#'3D-ResNets-PyTorch/results/UCF101_spacebagnet'
if model == 'resnet':
    result_path += "_" + str(model_depth)
else:
    result_path += "_" + str(receptive_size)


if pretrained:
    result_path += "_kinetics_pretrained"
    text += " --pretrain_path=" + pretrain_path
    text += " --n_pretrain_classes=" + str(n_pretrain_classes)
    
index = str(len([f for f in os.listdir(root_path+results_root) if result_path in f]))
result_path += "_" + index

text += " --result_path=" + results_root + result_path

print(text)

if submit_on_cluster:
    with open(result_path+".sbatch", "w") as file:
        file.write(text)
    os.system("sbatch "+result_path+".sbatch")    
else:
    os.system(text)

