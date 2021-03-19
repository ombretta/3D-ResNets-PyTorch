#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:39:58 2021

@author: ombretta
"""

import os 
import numpy as np
import csv


class model_parameters:
    
    def __init__(self, result_directory, datasets_list, models_list, 
                 datasets_info, dataset='', model='resnet', 
                 model_size='50', num_frames='48', t_stride='1', size='68', 
                 bs='8'):
       self.result_directory = result_directory
       self.dataset = self.get_configs(result_directory, datasets_list, dataset)
       self.model = self.get_configs(result_directory, models_list, model)
       self.model_size, self.receptive_size = self.get_model_size(result_directory, self.model)
       self.num_frames = self.get_input_parameters_values(r, 'frames', default=num_frames)
       self.t_stride = self.get_input_parameters_values(r, 'stride', default=t_stride)
       self.size = self.get_input_parameters_values(r, 'size', default=size)
       self.bs = self.get_input_parameters_values(r, 'bs', default=bs)
       self.pretrained = True if "pretrained" in result_directory else False
       self.num_run = result_directory.split("_")[-1]
       self.root_path, self.dataset_path, self.annotation_file, \
           self.file_type, self.value_scale, self.num_classes, \
           self.inference_subset = self.get_dataset_info(datasets_info, self.dataset)

    def get_configs(self, dir_name, configs_list, default=""):
        dir_name = "".join(dir_name.split("_")[:5])
        config = [d for d in configs_list if d in dir_name.lower()]
        if len(config) == 0: return default
        config = config[np.argmax([len(d) for d in config])]
        return config
    
    def get_model_size(self, r, model):
        if model not in r: return '50'
        size = r.split(model+"_")[1].split("_")[0]
        if "bagnet" in model: model_size, receptive_size = '50', size 
        else: model_size, receptive_size = size, '9' 
        return model_size, receptive_size
    
    def get_input_parameters_values(self, r, parameter, default=""):
        if parameter not in r: return default
        token = [f for f in r.split("_") if parameter in f][0]
        value = [i for i in token.split(parameter) if i != ''][0]
        return value

    def get_dataset_info(self, datasets_info, dataset_name):
        return list(datasets_info[dataset_name].values())
    
    
def decode(text):
    if '\ufeff' in text:
        text = text.replace('\ufeff', '')
    return text

def read_dataset_info(file_path):
    datasets_info = {}
    with open(file_path, "r", encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONE)
        count = 0
        attributes = []
        for row in reader:
            row = [decode(r) for r in row]
            print(row)
            
            if count == 0: attributes = row
            else: 
                dataset_name = row[0]
                dataset_info = {}
                for i in range(1, len(attributes)):
                    dataset_info[attributes[i]] = row[i]
                datasets_info[dataset_name] = dataset_info
            count += 1
    return datasets_info        
        

#%%
        
res_root = "results/"

datasets = ['kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']
dataset_paths = {'kinetics' : 'CV-DataSets/kinetics400/', }
models = ['resnet', 'vidbagnet', 'vidbagnet_tem']
datasets_info_file = "datasets_info.csv"
datasets_info = read_dataset_info(datasets_info_file)
# print(datasets_info)

results_dirs = [d for d in os.listdir(res_root) if os.path.exists(res_root+d+"/opts.json")]
print(results_dirs)
for r in results_dirs[:1]:
    print(r)
    
    model_configs = model_parameters(r, datasets, models, datasets_info)
    
    # print(model_configs.dataset)
    # print(model_configs.model)
    # print(model_configs.model_size)
    # print('num_frames', model_configs.num_frames, 't_stride',\
    #       model_configs.t_stride, 'size', model_configs.size,\
    #           'bs', model_configs.bs)
    # print("num_run", model_configs.num_run)
    checkpoints = [f for f in os.listdir(res_root+r) if "pth" in f]
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    for c in checkpoints[:1]:
        
        epoch  = c.split("_")[1].split(".")[0]
        print(c, epoch)
        input_text = "python main.py --root_path=" + model_configs.root_path + \
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
            " --tensorboard --ft_begin_module=" + res_root+r+"/"+c + \
            " --result_path=" + res_root+r + \
            " --no_train --no_val --inference"
    
        os.system(input_text)
    
    
    
