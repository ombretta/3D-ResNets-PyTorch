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
    
    
    model_results = {}
    
    for c in checkpoints[round(len(checkpoints)/2):]:
        
        epoch  = c.split("_")[1].split(".")[0]
        print(c, epoch)
        input_text = "--root_path=" + model_configs.root_path + \
            " --video_path=" + model_configs.dataset_path + \
            " --annotation_path=" + model_configs.annotation_file + \
            " --dataset=" + model_configs.dataset + \
            " --n_classes=" + model_configs.num_classes + \
            " --sample_size=" + model_configs.size + \
            " --sample_duration=" + model_configs.num_frames + \
            " --sample_t_stride=" + model_configs.t_stride + \
            " --train_crop=center --train_t_crop=random " + \
            " --value_scale=" + model_configs.value_scale + \
            " --inference_batch_size=1 " + \
            " --inference_subset=" + model_configs.inference_subset + \
            " --inference_stride=2 --inference_crop=center --n_threads=4 " + \
            " --model=" + model_configs.model + \
            " --model_depth=" + model_configs.model_size + \
            " --receptive_size=" + model_configs.receptive_size + \
            " --output_topk=1 --file_type=" + model_configs.file_type + \
            " --tensorboard --ft_begin_module=3D-ResNets-PyTorch/" + res_root+r+"/"+c + \
            " --result_path=3D-ResNets-PyTorch/" + res_root+r + \
            " --no_train --no_val --inference" + \
            " --n_pretrain_classe=s" + model_configs.num_classes + \
            " --pretrain_path=3D-ResNets-PyTorch/" + res_root+r+"/"+c 
            
        opt = get_opt(arguments_string = input_text)
    
        opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
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
