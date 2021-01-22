#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:16:11 2021

@author: ombretta
"""

import argparse
import json
from pathlib import Path
import os

import pandas as pd

from .utils import get_n_frames, get_n_frames_hdf5


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        basename = '%s_%s_%s' % (row['youtube_id'], '%06d' % row['time_start'],
                                 '%06d' % row['time_end'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database


def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()


def convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 video_dir_path, video_type, dst_json_path):
    train_videos_path = os.path.join(video_dir_path, "h5_train_frames")
    val_videos_path = os.path.join(video_dir_path, "h5_valid_frames")
    
    labels = os.listdir(train_videos_path)
    labels = [l for l in labels if os.path.isdir(os.path.join(train_videos_path, l))]
    
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    
    for label in labels:
        
        train_videos = os.listdir(os.path.join(train_videos_path, label))
        valid_videos = os.listdir(os.path.join(val_videos_path, label))
        
        for video in train_videos:
            dst_data['database'][video]["subset"] = "training"
            dst_data['database'][video]['annotations']['label'] = label
            
            video_path = os.path.join(train_videos_path, label, video+".h5")
            if os.path.exists(video_path):
                n_frames = get_n_frames_hdf5(video_path)
                dst_data['database'][video]['annotations']['segment'] = (0, n_frames)
        
        for video in valid_videos:
            dst_data['database'][video]["subset"] = "validation"
            dst_data['database'][video]['annotations']['label'] = label
            
            video_path = os.path.join(val_videos_path, label, video+".h5")
            if os.path.exists(video_path):
                n_frames = get_n_frames_hdf5(video_path)
                dst_data['database'][video]['annotations']['segment'] = (0, n_frames)
                

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path',
                        default=None,
                        type=Path,
                        help=('Directory path including '
                              'kinetics_train.csv, kinetics_val.csv, '
                              '(kinetics_test.csv (optional))'))
    parser.add_argument(
        'n_classes',
        default=700,
        type=int,
        help='400, 600, or 700 (Kinetics-400, Kinetics-600, or Kinetics-700)')
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg or hdf5).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('video_type',
                        default='jpg',
                        type=str,
                        help=('jpg or hdf5'))
    parser.add_argument('dst_path',
                        default=None,
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    assert args.video_type in ['jpg', 'hdf5']

    train_csv_path = (args.dir_path /
                      'kinetics-{}_train.csv'.format(args.n_classes))
    val_csv_path = (args.dir_path /
                    'kinetics-{}_val.csv'.format(args.n_classes))
    test_csv_path = (args.dir_path /
                     'kinetics-{}_test.csv'.format(args.n_classes))

    convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 args.video_path, args.video_type,
                                 args.dst_path)
