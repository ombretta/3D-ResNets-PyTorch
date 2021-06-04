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

if not os.path.exists("plots/"): os.mkdir("plots/")

plots_name = "movingmnist_longterm_resnet50_vs_videobagnet_tem9_10videosxdigit"

res_dirs = [f for f in os.listdir("results/") if "mnist" in f]
discard_dirs = ["motion", "blackframes", "val_1tstride"]
res_dirs = [f for f in res_dirs if all([d not in f for d in discard_dirs])]

# Filter according to training setting
colors = ["red", "green", "orange", "black"]
filtering_criteria1 = ["resnet_50", "resnet_34", "resnet_18", "bagnet_tem"] #["32frames", "64frames", "128frames"]
filtering_criteria2 = [''] #["64frames"]
filtering_criteria3 = ["longterm"] #["32frames_8", "64frames_4", "128frames_2", "256"] #["256"]
filtering_criteria_annotation_path = [''] #["longterm/"]
filtering_criteria_sampling = ["even_crops"]
fig_train, ax_train = plt.subplots(figsize=(12, 8))
fig_val, ax_val = plt.subplots(figsize=(12, 8))

for r in res_dirs:
    
    if os.path.exists("results/"+r+"/opts.json"):
            with open("results/"+r+"/opts.json", "r") as f:
                opts = json.load(f)
            # print(opts["annotation_path"])
            # if "mnist_json_10.json" in opts["annotation_path"]: 
            #     print(r)
    
    low_data_regime = False
    
    if [f for f in os.listdir("results/"+r) if "events.out" in f] and \
        any([c in r for c in filtering_criteria1]) and \
        any([c in r for c in filtering_criteria2]) and \
        any([c in r for c in filtering_criteria3]):
            
        # print(r)
        event_acc = EventAccumulator("results/"+r)
        event_acc.Reload()
        # Show all tags in the log file
        # print(event_acc.Tags())
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        
        if os.path.exists("results/"+r+"/opts.json"):
            with open("results/"+r+"/opts.json", "r") as f:
                opts = json.load(f)
            # print(opts["annotation_path"])
            # if "mnist_json_10.json" in opts["annotation_path"]: 
            #     print(r)
            #     low_data_regime = True
            # print(opts)
        

        if (True or low_data_regime) and \
            any([c in opts['annotation_path'] for c in filtering_criteria_annotation_path]) \
            and any ([c in opts['train_t_crop'] for c in filtering_criteria_sampling]):
            if event_acc.scalars.Keys() != []:
                train_losses, train_epochs, train_accs = zip(*event_acc.Scalars('train/acc'))
                val_losses, val_epochs, val_accs = zip(*event_acc.Scalars('val/acc'))
                if len(val_losses) <10:
                    os.system("rm -r results/"+r)
                if len(val_losses) >= 1: # or train_accs[-1] > 0.95:
                    
                    # print(len(val_losses))
                    print(r)
                    print("train", round(np.max(train_accs)*100, 2), np.argmax(train_accs), \
                          "val", round(np.max(val_accs)*100, 2), np.argmax(val_accs))
                    
                    if os.path.exists("results/"+r+"/checkpoints_test_results.json"):
                        with open("results/"+r+"/checkpoints_test_results.json", "r") as f:
                            test = json.load(f)
                        print(test)
                        
                    color = [colors[i] for i in range(len(filtering_criteria1)) \
                              if [c in r for c in filtering_criteria1][i]][0]
                    
                    # color = [colors[i] for i in range(len(filtering_criteria_annotation_path)) \
                    #           if [c in opts['annotation_path'] for c in filtering_criteria_annotation_path][i]][0]
                        
                    frequency = [c in opts['annotation_path'] for c in filtering_criteria1][0]
                    linestyle='dashed' if low_data_regime else 'solid'
                    if np.max(val_accs)*100 > 1:
                        ax_train.plot(train_epochs, [t*100 for t in train_accs], 
                                      label=r+str(frequency), color=color, 
                                      linewidth=2, linestyle=linestyle)
                        ax_train.set_title("Training accuracies")
                        ax_train.set_xlabel("Epochs")        
                        ax_train.set_ylabel("Train acc (%)")        
                        ax_train.legend()
                        
                        ax_val.plot(val_epochs, [v*100 for v in val_accs],
                                    label=r, color=color, 
                                    linewidth=2, linestyle=linestyle)
                        ax_val.set_title("Validation accuracies")
                        ax_val.set_xlabel("Epochs")        
                        ax_val.set_ylabel("Val acc (%)")   
                        ax_val.legend()

# fig_train.savefig("plots/"+plots_name+"_train.png")
# fig_val.savefig("plots/"+plots_name+"_val.png")