#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:53:54 2021

@author: ombretta
"""


import json
import os

import cv2
from .utils import get_n_frames, get_n_frames_hdf5

import torch
import pickle as pkl


def load_data(filename, with_torch = False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else: print("File", filename, "does not exists.")        
                

def get_video_properties(cap):
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return frameCount, frameWidth, frameHeight, fps


def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frameCount, _, _, _ = get_video_properties(cap)
    cap.release()
    return frameCount


def video_dictionary(video_path, subset, label):
    video_dict = {}
    video_dict["subset"] = subset
    video_dict['annotations'] = {}
    video_dict['annotations']['label'] = label
    
    if os.path.exists(video_path):
        print(video_path)
        n_frames = get_num_frames(video_path)
        print(n_frames)
        video_dict['annotations']['segment'] = (0, n_frames)
    return video_dict


if __name__ == '__main__':
    
    videos_path = "/tudelft.net/staff-bulk/ewi/insy/VisionLab/ombrettastraff/"+\
        "instructional_videos/experiments/breakfast/BreakfastII_15fps_qvga_sync/"
    
    train_videos = load_data("/tudelft.net/staff-bulk/ewi/insy/VisionLab/"+\
                        "ombrettastraff/instructional_videos/i3d_breakfast/"+\
                        "data/processed/video_sets_split/train_videos.dat")
        
    valid_videos = load_data("/tudelft.net/staff-bulk/ewi/insy/VisionLab/"+\
                        "ombrettastraff/instructional_videos/i3d_breakfast/"+\
                        "data/processed/video_sets_split/valid_videos.dat")
    
    labels = load_data("/tudelft.net/staff-bulk/ewi/insy/VisionLab/"+\
                        "ombrettastraff/instructional_videos/i3d_breakfast/"+\
                        "data/processed/classes_labels.dat")
    
    labels = list(labels.keys())
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
        
    
    for video_folder in [f for f in os.listdir(videos_path) 
                         if os.path.isdir(videos_path+f)]:
        print(video_folder)
        
        for cameras in [c for c in os.listdir(videos_path+video_folder) 
                        if os.path.isdir(videos_path+video_folder+"/"+c)]:
            print(cameras)
            
            videos = [v for v in os.listdir(videos_path+video_folder+"/"+cameras) 
                      if ".mp4" in v and "resized" not in v]
            for original_video in videos:
                print(original_video)
                
                video_path = videos_path + video_folder + "/" + cameras + "/" + original_video
                video_name = cameras+"_"+original_video[:-4]
                video_label = original_video.split("_")[1].split(".")[0]
                
                subset = "training" if video_name in train_videos else "validation"
                
                dst_data['database'][video_name] = video_dictionary(video_path, subset, video_label)
    
    dst_json_path = "/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/breakfast/breakfast.json"            
    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)