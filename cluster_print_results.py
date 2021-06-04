#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:00:15 2021

@author: ombretta
"""

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from matplotlib import pyplot as plt
import json
import sys


def main(dirs_dataset_filter="mnist", more_discard_dirs=[], 
         filtering_criteria_models=["resnet_50", "resnet_34", "resnet_18", 
        "bagnet_tem_9", "bagnet_tem_17"], filtering_criteria_frames=[],
        filtering_criteria_others=[], filtering_criteria_annotation_path=[],
        filtering_criteria_sampling=["center"]):
    
    
    res_dirs = [f for f in os.listdir("results/") if dirs_dataset_filter in f]
    discard_dirs = ["motion", "blackframes", "val_1tstride"] + more_discard_dirs
    res_dirs = [f for f in res_dirs if all([d not in f for d in discard_dirs])]
    
    for r in res_dirs:
        
        # if os.path.exists("results/"+r+"/opts.json"):
        #         with open("results/"+r+"/opts.json", "r") as f:
        #             opts = json.load(f)
        #         print(opts["annotation_path"])
        
        if [f for f in os.listdir("results/"+r) if "events.out" in f] and \
            any([c in r for c in filtering_criteria_models]) and \
            any([c in r for c in filtering_criteria_frames]) and \
            any([c in r for c in filtering_criteria_others]) and \
            os.path.exists("results/"+r+"/opts.json"):
                
            print(r)
                
            event_acc = EventAccumulator("results/"+r)
            event_acc.Reload()
            
            with open("results/"+r+"/opts.json", "r") as f:
                opts = json.load(f)            
            
            print(opts['annotation_path'])
    
            if any([c in opts['annotation_path'] for c in filtering_criteria_annotation_path]) \
                and any ([c in opts['train_t_crop'] for c in filtering_criteria_sampling]):
                    
                if event_acc.scalars.Keys() != []:
                    train_losses, train_epochs, train_accs = zip(*event_acc.Scalars('train/acc'))
                    val_losses, val_epochs, val_accs = zip(*event_acc.Scalars('val/acc'))
                    if len(val_losses) <10:
                        os.system("rm -r results/"+r)
                    if len(val_losses) >= 10 or train_accs[-1] > 0.95:
                        
                        # print(len(val_losses))
                        print(r)
                        print("train", round(np.max(train_accs)*100, 2), np.argmax(train_accs), \
                              "val", round(np.max(val_accs)*100, 2), np.argmax(val_accs))
                        
                        if os.path.exists("results/"+r+"/checkpoints_test_results.json"):
                            with open("results/"+r+"/checkpoints_test_results.json", "r") as f:
                                test = json.load(f)
                            print(test)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    
    parser.add_argument('--discard_dirs', type=str, dest='discard_dirs', default="")
    parser.add_argument('--dirs_dataset_filter', type=str, dest='dirs_dataset_filter', default="mnist")
    parser.add_argument('--filtering_criteria_models', type=str, dest='filtering_criteria_models', default="")
    parser.add_argument('--filtering_criteria_frames', type=str, dest='filtering_criteria_frames', default="")
    parser.add_argument('--filtering_criteria_others', type=str, dest='filtering_criteria_others', default="")
    parser.add_argument('--filtering_criteria_annotation_path', type=str, dest='filtering_criteria_annotation_path', default="")
    parser.add_argument('--filtering_criteria_sampling', type=str, dest='filtering_criteria_sampling', default="")
    
    args = parser.parse_args(sys.argv[1:])
    args.discard_dirs = args.discard_dirs.split(",")
    args.filtering_criteria_models = args.filtering_criteria_models.split(",")
    args.filtering_criteria_frames = args.filtering_criteria_frames.split(",")
    args.filtering_criteria_others = args.filtering_criteria_others.split(",")
    args.filtering_criteria_annotation_path = args.filtering_criteria_annotation_path.split(",")
    args.filtering_criteria_sampling = args.filtering_criteria_sampling.split(",")
    
    main(**{k: v for (k, v) in vars(args).items() if v is not None})