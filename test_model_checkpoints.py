#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:21:25 2021

@author: ombretta
"""

import os 
import sys
from main import get_opt, main_worker
import torch
from torch.backends import cudnn
import json 
import torchvision

from test_all import model_parameters, read_dataset_info


def main(r, res_root):
    datasets = ['kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
            'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
            'movingmnist_longterm']
    models = ['resnet', 'vidbagnet', 'vidbagnet_tem']
    datasets_info_file = "datasets_info.csv"
    datasets_info = read_dataset_info(datasets_info_file)
    
    model_configs = model_parameters(r, datasets, models, datasets_info)
        
    print(model_configs.dataset)
    print(model_configs.model)
    print(model_configs.model_size)
    print('num_frames', model_configs.num_frames, 't_stride',\
          model_configs.t_stride, 'size', model_configs.size,\
              'bs', model_configs.bs)
    print("num_run", model_configs.num_run)
    checkpoints = [f for f in os.listdir(res_root+r) if "pth" in f]
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    
    '''{"root_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff", 
    "video_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/movingMNIST/movingmnistdata", 
    "annotation_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/movingMNIST/movingmnistdata/mnist_json.json", 
    "result_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/3D-ResNets-PyTorch/results/movingmnist_resnet_18_32frames_32size_bs16_1", 
    "dataset": "movingmnist", "n_classes": 10, "n_pretrain_classes": 10,
    "pretrain_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/3D-ResNets-PyTorch/results/movingmnist_resnet_18_32frames_32size_bs16_1/save_100.pth", 
    "ft_begin_module": "3D-ResNets-PyTorch/results/movingmnist_resnet_18_32frames_32size_bs16_1/save_100.pth", 
    "sample_size": 32, "sample_duration": 32, "sample_t_stride": 1, "train_crop": "center", "train_crop_min_scale": 0.25, 
    "train_crop_min_ratio": 0.75, "no_hflip": false, "colorjitter": false, "train_t_crop": "random", "learning_rate": 0.1, 
    "momentum": 0.9, "dampening": 0.0, "weight_decay": 0.001, "mean_dataset": "kinetics", "no_mean_norm": false, "no_std_norm": 
        false, "value_scale": 1, "nesterov": false, "optimizer": "sgd", "lr_scheduler": "multistep", "multistep_milestones": [50, 100, 150], 
        "overwrite_milestones": false, "plateau_patience": 10, "batch_size": 128, "inference_batch_size": 1, 
        "batchnorm_sync": false, "n_epochs": 200, "n_val_samples": 3, 
        "resume_path": "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/3D-ResNets-PyTorch/results/movingmnist_resnet_18_32frames_32size_bs16_1/save_100.pth", 
        "no_train": true, "no_val": true, "inference": true, "inference_subset": "test", "inference_stride": 2, 
        "inference_crop": "center", "inference_no_average": false, "no_cuda": false, "n_threads": 4, 
        "checkpoint": 10, "model": "resnet", "model_depth": 18, "receptive_size": 9, "conv1_t_size": 7,
        "conv1_t_stride": 1, "no_max_pool": false, "resnet_shortcut": "B", "resnet_widen_factor": 1.0, 
        "wide_resnet_k": 2, "resnext_cardinality": 32, "input_type": "rgb", "manual_seed": 1, 
        "accimage": false, "output_topk": 1, "file_type": "jpg", "tensorboard": true,
        "distributed": false, "dist_url": "tcp://127.0.0.1:23456", "world_size": -1, 
        "n_finetune_classes": 10, "arch": "resnet-18", "begin_epoch": 1, "mean": [0.4345, 0.4051, 0.3775], 
        "std": [0.2768, 0.2713, 0.2737], "n_input_channels": 3}'''
    
    model_results = {}
    
    if os.path.exists(os.path.join(res_root, r, "/opts.json")):
        with open(os.path.join(res_root, r, "/opts.json"), "r") as f:
            model_opts = json.load(f)
    
        for c in checkpoints[round(len(checkpoints)/2):]:
            
            epoch  = c.split("_")[1].split(".")[0]
            print(c, epoch)
            input_text = "--root_path=" + model_opts.root_path + \
                " --video_path=" + model_opts.video_path + \
                " --annotation_path=" + model_opts.annotation_path + \
                " --dataset=" + model_opts.dataset + \
                " --n_classes=" + model_opts.n_classes + \
                " --sample_size=" + model_opts.sample_size + \
                " --sample_duration=" + model_opts.sample_duration + \
                " --sample_t_stride=" + model_opts.sample_t_stride + \
                " --train_crop=" + model_opts.train_crop + \
                " --train_t_crop=" + model_opts.train_t_crop + \
                " --value_scale=" + model_opts.value_scale + \
                " --inference_batch_size=1 " + \
                " --inference_subset=" + model_opts.inference_subset + \
                " --inference_stride=" + model_opts.sample_t_stride + \
                " --inference_crop=" + model_opts.train_crop + \
                " --n_threads=4 " + \
                " --model=" + model_opts.model + \
                " --model_depth=" + model_opts.model_depth + \
                " --receptive_size=" + model_opts.receptive_size + \
                " --output_topk=1 --file_type=" + model_opts.file_type + \
                " --ft_begin_module=3D-ResNets-PyTorch/" + res_root+r+"/"+c + \
                " --result_path=" + model_opts.result_path + \
                " --no_train --no_val --inference" + \
                " --n_pretrain_classes=" + model_opts.n_classes + \
                " --pretrain_path=3D-ResNets-PyTorch/" + res_root+r+"/"+c + \
                " --resume_path=3D-ResNets-PyTorch/" + res_root+r+"/"+c
                
            opt = get_opt(arguments_string = input_text, save=False)
        
            opt.device = torch.device('cpu' if model_configs.no_cuda else 'cuda')
            if not opt.no_cuda:
                cudnn.benchmark = True
            if opt.accimage:
                torchvision.set_image_backend('accimage')
        
            opt.ngpus_per_node = torch.cuda.device_count()
            inference_results = main_worker(-1, opt)
            
            model_results['epoch_'+epoch] = inference_results
        
        with open(res_root+r+"/checkpoints_test_results.json", "w") as f:
            json.dump(model_results, f)
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--r', type=str, dest='r', default='') # name of the results folder for the tested model
    parser.add_argument('--res_root', type=str, dest='res_root', default='') # Location of the results directories
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
