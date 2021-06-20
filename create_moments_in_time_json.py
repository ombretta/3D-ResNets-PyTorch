#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:52:17 2021

@author: ombretta
"""

import json 
import os 
import sys
import cv2

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frameCount, frameWidth, frameHeight, fps

def get_classes(file):
    with open(file, "r") as f:
        text = f.readlines()
    print(text)
    classes = {}
    for row in text:
        print(row)
        if "," in row:
            classes[row.split(",")[0]] = row.split(",")[1].replace("\n", "")
    print(classes)
    return classes

def main(dataset_dir="../../../CV-DataSets/Moments_in_Time_Raw/"):
        
    json_file = os.path.join(dataset_dir, "moments_in_time.json")
    print("dataset folder:", dataset_dir)
    
    # Get classes
    moments_categories_file = dataset_dir+"moments_categories.txt"
    #moments_categories_file = "/Users/ombretta/Desktop/moments_categories.txt" # to delete
    classes = get_classes(moments_categories_file)
    
    
    # Filling the dataset dictionary
    mom_in_time = {}
    mom_in_time['labels'] = list(classes.keys())
    mom_in_time['database'] = {}
    
    # Training
    training_path = dataset_dir+"training/"
    subset = "training"
    
    for video_class in os.listdir(training_path):
        
        if os.path.isdir(os.path.join(training_path, video_class)):
            
            label = video_class
            print(video_class, label)
            
            for v in os.listdir(os.path.join(training_path, video_class)):
                
                video_path = os.path.join(training_path, video_class, v)
                tot_frames, _, _, _ = get_video_properties(video_path)
                segment =  [0, tot_frames]
                
                video_name = v.split(".mp4")[0]
                print(video_name, subset, label, tot_frames)
                if tot_frames > 0:
                    mom_in_time['database'][video_name] = {}
                    mom_in_time['database'][video_name]['subset'] = subset
                    mom_in_time['database'][video_name]['annotations'] = {'label': label, 'segment': segment}
                else: 
                    print("skipped")
            
    # Validation
    validation_path = dataset_dir+"validation/"
    subset = "validation"
    
    for video_class in os.listdir(validation_path):
        
        if os.path.isdir(os.path.join(validation_path, video_class)):
            
            label = video_class
            print(video_class, label)
            
            for v in os.listdir(os.path.join(validation_path, video_class)):
                
                video_path = os.path.join(validation_path, video_class, v)
                tot_frames, _, _, _ = get_video_properties(video_path)
                segment =  [0, tot_frames]

                video_name = v.split(".mp4")[0]                
                print(video_name, subset, label, tot_frames)
                if tot_frames > 0:
                    mom_in_time['database'][video_name] = {}
                    mom_in_time['database'][video_name]['subset'] = subset
                    mom_in_time['database'][video_name]['annotations'] = {'label': label, 'segment': segment}
                else: 
                    print("skipped")
                
        
    with open(json_file, "w") as file: 
        json.dump(mom_in_time, file)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dataset_dir', type=str, dest='dataset_dir')
    args = parser.parse_args(sys.argv[1:])
    main(**{k: v for (k, v) in vars(args).items() if v is not None})
