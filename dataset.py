from torchvision import get_image_backend

from datasets.videodataset import VideoDataset
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from datasets.activitynet import ActivityNet
from datasets.breakfast import get_breakfast_dataset, load_videos_sets
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5

import os
from pathlib import Path

def usual_image_name_formatter(x):
    return f'image_{x:05d}.jpg'

def mnist_image_name_formatter(x):
    return f'{x:d}.jpg'


def get_training_data(video_path,
                      annotation_path,
                      dataset_name,
                      input_type,
                      file_type,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None, 
                      sample_t_stride=1):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', None]

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'
        
        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            image_name_formatter = mnist_image_name_formatter
        else: image_name_formatter = usual_image_name_formatter

        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
        
        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)

    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
            
        if dataset_name in ['kinetics', 'mini_kinetics']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}')
        else:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.hdf5')
    print("video_path_formatter", video_path_formatter)

    if dataset_name == 'activitynet':
        training_data = ActivityNet(video_path,
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)
        
    elif dataset_name in ['kinetics', 'mini_kinetics']:
        training_data = VideoDataset(Path(os.path.join(video_path,"h5_train_frames")),
                                    annotation_path,
                                    'training',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform,
                                    target_transform=target_transform,
                                    video_loader=loader,
                                    video_path_formatter=video_path_formatter)
    
    else:
        print("Building VideoDataset for", dataset_name)
        print(spatial_transform)
        print(temporal_transform)
        print(loader)
        training_data = VideoDataset(video_path,
                                     annotation_path,
                                     'training',
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter)

    return training_data


def get_validation_data(video_path,
                        annotation_path,
                        dataset_name,
                        input_type,
                        file_type,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None, 
                        sample_t_stride=1):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', None]

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            image_name_formatter = mnist_image_name_formatter
        else: image_name_formatter = usual_image_name_formatter
        
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)

        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
            
        if dataset_name in ['kinetics', 'mini_kinetics']:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}')
        else:
            video_path_formatter = (lambda root_path, label, video_id: root_path /
                                    label / f'{video_id}.hdf5')

    if dataset_name == 'activitynet':
        validation_data = ActivityNet(video_path,
                                      annotation_path,
                                      'validation',
                                      spatial_transform=spatial_transform,
                                      temporal_transform=temporal_transform,
                                      target_transform=target_transform,
                                      video_loader=loader,
                                      video_path_formatter=video_path_formatter)
        
    elif dataset_name in ['kinetics', 'mini_kinetics']:
        validation_data = VideoDatasetMultiClips(
                            Path(os.path.join(video_path,"h5_valid_frames")),
                            annotation_path,
                            'validation',
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform,
                            video_loader=loader,
                            video_path_formatter=video_path_formatter)
        
    else:
        validation_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter)

    return validation_data, collate_fn


def get_inference_data(video_path,
                       annotation_path,
                       dataset_name,
                       input_type,
                       file_type,
                       inference_subset,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None, 
                        sample_t_stride=1):
    assert dataset_name in [
        'kinetics', 'mini_kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit', 
        'breakfast', 'mini_breakfast', 'movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm'
    ]
    assert input_type in ['rgb', 'flow']
    assert file_type in ['jpg', 'hdf5', None]
    assert inference_subset in ['train', 'val', 'test']

    if file_type == 'jpg':
        assert input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            image_name_formatter = mnist_image_name_formatter
        else: image_name_formatter = usual_image_name_formatter
        
        if get_image_backend() == 'accimage':
            from datasets.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)

        if dataset_name in ['movingmnist', 'movingmnist_blackframes',
        'movingmnist_longterm']:
            video_path_formatter = (
            lambda root_path, label, video_id: root_path / video_id)
    else:
        if input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path /
                                label / f'{video_id}.hdf5')

    if inference_subset == 'train':
        subset = 'training'
    elif inference_subset == 'val':
        subset = 'validation'
    elif inference_subset == 'test':
        subset = 'testing'
    if dataset_name == 'activitynet':
        inference_data = ActivityNet(video_path,
                                     annotation_path,
                                     subset,
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     target_transform=target_transform,
                                     video_loader=loader,
                                     video_path_formatter=video_path_formatter,
                                     is_untrimmed_setting=True)
        
    else:
        inference_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            subset,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
            target_type=['video_id', 'segment'])

    return inference_data, collate_fn
