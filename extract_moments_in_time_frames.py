#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:39:25 2021

@author: ombretta
"""

import cv2
import os

dataset_path = "/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/Moments_in_Time_Raw/"

for data_set in ["training", "validation"]:
    
    dest = dataset_path+data_set+"_jpg/"
    os.mkdir(dest)
    
    for video_class in os.listdir(dataset_path+data_set):
        
        os.mkdir(dest+video_class)
        
        for v in os.listdir(dataset_path+data_set+"/"+video_class):
            
            video_path = dataset_path+data_set+"/"+video_class+"/"+v
            video_dest = dest + video_class + "/" + v.split(".mp4")[0]
            
            print("Saving", video_dest)
            
            os.mkdir(video_dest)
            os.chdir(video_dest)
            
            vidcap = cv2.VideoCapture(video_path)
            success,image = vidcap.read()
            count = 0
            while success:
              cv2.imwrite("image_%d.jpg" % count, image)     # save frame as JPEG file      
              success,image = vidcap.read()
              print('Read a new frame: ', success)
              count += 1
