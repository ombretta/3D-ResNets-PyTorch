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

if not os.path.exists("plots/"): os.mkdir("plots/")

res_dirs = [f for f in os.listdir("results/") if "mnist_longterm_resnet" in f]# if \
            # "events.out.tfevents.1615832673.ewi2.hpc.tudelft.nl" \
            #     in os.listdir("results/"+f)]

fig = plt.figure(figsize=(16, 8))
for r in res_dirs:
    print(r)
    if [f for f in os.listdir("results/"+r) if "events.out" in f]:
        event_acc = EventAccumulator("results/"+r)
        event_acc.Reload()
        # Show all tags in the log file
        # print(event_acc.Tags())
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        
        train_losses, train_epochs, train_accs = zip(*event_acc.Scalars('train/acc'))
        val_losses, val_epochs, val_accs = zip(*event_acc.Scalars('val/acc'))
        print("train", round(np.max(train_accs)*100, 2), np.argmax(train_accs), \
              "val", round(np.max(val_accs)*100, 2), np.argmax(val_accs))
        
        if np.max(val_accs)*100 > 50:
            plt.plot(val_epochs, val_accs, label=r, linewidth=2)
            plt.xlabel("Epochs")        
            plt.xlabel("Val acc (%)")        
            plt.legend()

# plt.savefig("plots/mnist_vidbagnet_tem.png")