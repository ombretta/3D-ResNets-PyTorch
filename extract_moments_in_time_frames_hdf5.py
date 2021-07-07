#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:55:57 2021

@author: ombretta
"""

import cv2
import os
import h5py
import numpy as np


def extract_frames(video_path, video_dest):
    with h5py.File(video_dest, "w") as h5_file:
                   
        cap = cv2.VideoCapture(video_path)
        frameCount, frameWidth, frameHeight, fps = get_video_properties(cap)
        print(video_path, frameCount, frameWidth, frameHeight, fps)
        
        # Read all video frames 
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))
        
        fc = 0
        ret = True
        
        while ((fc < frameCount)  and ret):
            ret, frame_bgr = cap.read()
            frame_rgb = frame_bgr[:, :, [2, 1, 0]] # Swap bgr to rgb
            buf[fc] = frame_rgb[:,:,:] 
            fc += 1
            
        cap.release()
        
        h5_file.create_dataset('video', data=frame_rgb)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_images = h5_file['video'][...].shape[0]
        print(num_frames, num_images)
    return num_frames == num_images


def get_video_properties(cap):
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return frameCount, frameWidth, frameHeight, fps


def main():
    dataset_path = "/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/Moments_in_Time_Raw/"
    
    for data_set in ["training", "validation"]:
            
        dest = dataset_path+data_set+"_h5/"
        if not os.path.exists(dest):
            os.mkdir(dest)
        
        for video_class in os.listdir(dataset_path+data_set):
            
            if not os.path.exists(dest+video_class):
                os.mkdir(dest+video_class)
            
            for v in os.listdir(dataset_path+data_set+"/"+video_class):
                
                video_path = dataset_path+data_set+"/"+video_class+"/"+v
                video_dest = dest + video_class + "/" + v.split(".mp4")[0] + ".hdf5"
                
                if os.path.exists(video_dest):
                    print(video_dest, "already exists")
                else:
                
                    print("Saving", video_dest)
                    print(extract_frames(video_path, video_dest))
                

if __name__== "__main__":
    main()
