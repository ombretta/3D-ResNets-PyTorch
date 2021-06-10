#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:54:05 2021

@author: ombretta
"""

import os 
import sys
from main import get_opt, main_worker
import torch
from torch.backends import cudnn
import json 
import torchvision


def main(r, res_root, checkpoint_epoch=0, test_data_path='', annotation_path = ''):
    
    checkpoint = "save_"+str(checkpoint_epoch)+".pth"
        
    if os.path.exists(os.path.join(res_root, r, "opts.json")):
        with open(os.path.join(res_root, r, "opts.json"), "r") as f:
            model_opts = json.load(f)

        print("Testing", r, checkpoint)
        
        if test_data_path == '': test_data_path = model_opts["video_path"]
        if annotation_path == '': annotation_path = model_opts["annotation_path"]
        print("Test set:", test_data_path, annotation_path)
        
        input_text = "--root_path=" + model_opts["root_path"] + \
            " --video_path=" + test_data_path + \
            " --annotation_path=" + annotation_path + \
            " --dataset=" + model_opts["dataset"] + \
            " --n_classes=" + str(model_opts["n_classes"]) + \
            " --sample_size=" + str(model_opts["sample_size"]) + \
            " --sample_duration=" + str(model_opts["sample_duration"]) + \
            " --sample_t_stride=" + str(model_opts["sample_t_stride"]) + \
            " --train_crop=" + model_opts["train_crop"] + \
            " --train_t_crop=" + model_opts["train_t_crop"] + \
            " --value_scale=" + str(model_opts["value_scale"]) + \
            " --inference_batch_size=1 " + \
            " --inference_subset=" + model_opts["inference_subset"] + \
            " --inference_stride=" + str(model_opts["sample_t_stride"]) + \
            " --inference_crop=" + model_opts["train_crop"] + \
            " --n_threads=4 " + \
            " --model=" + model_opts["model"] + \
            " --model_depth=" + str(model_opts["model_depth"]) + \
            " --receptive_size=" + str(model_opts["receptive_size"]) + \
            " --output_topk=1 --file_type=" + model_opts["file_type"] + \
            " --ft_begin_module=3D-ResNets-PyTorch/" + res_root+r+"/"+checkpoint + \
            " --result_path=" + model_opts["result_path"] + \
            " --no_train --no_val --inference" + \
            " --n_pretrain_classes=" + str(model_opts["n_classes"]) + \
            " --pretrain_path=3D-ResNets-PyTorch/" + res_root+r+"/"+checkpoint + \
            " --resume_path=3D-ResNets-PyTorch/" + res_root+r+"/"+checkpoint
            
        opt = get_opt(arguments_string = input_text, save=False)
    
        opt.device = torch.device('cpu' if model_opts["no_cuda"] else 'cuda')
        if not opt.no_cuda:
            cudnn.benchmark = True
        if opt.accimage:
            torchvision.set_image_backend('accimage')
    
        opt.ngpus_per_node = torch.cuda.device_count()
        inference_results = main_worker(-1, opt)
        
        print(r, checkpoint, 'test acc', inference_results)
        print("Test set:", test_data_path, annotation_path)
        
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--r', type=str, dest='r', default='') # name of the results folder for the tested model
    parser.add_argument('--res_root', type=str, dest='res_root', default='') # Location of the results directories
    parser.add_argument('--checkpoint_epoch', type=int, dest='checkpoint_epoch', default=0) # Epoch of the checkpoint to test
    parser.add_argument('--test_data_path', type=str, dest='test_data_path', default='') # Location of the test set
    parser.add_argument('--annotation_path', type=str, dest='annotation_path', default='') # Location of the test annotations
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
