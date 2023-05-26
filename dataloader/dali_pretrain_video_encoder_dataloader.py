import random

import numpy as np
import torch
from nvidia.dali.pipeline import Pipeline, pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import hydra
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

@pipeline_def
def PreTrainVideoDataPipeline(folderpath, folderpath2, sequence_length=150, window_size =8, first_resize=128, second_resize=84 ):
    # batch, length, w, h, channel
    video_raw, labels = fn.readers.video(device="gpu", file_root=folderpath, sequence_length=sequence_length//window_size,
                                     random_shuffle=False, initial_fill=33, stride=8, name="video_replay_reader")

    video_raw_two, labels = fn.readers.video(device="gpu", file_root=folderpath2,
                                         sequence_length=sequence_length // window_size,
                                         random_shuffle=False, initial_fill=33, stride=8, name="video_replay_reader2")


    video= fn.resize(video_raw, size=[first_resize, first_resize])
    video_target = fn.resize(video_raw_two, size=[second_resize, second_resize])

    video = fn.crop_mirror_normalize(video, mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
                                      std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], device = "gpu",
                                     output_layout='FHWC')

    video_target = fn.crop_mirror_normalize(video_target, mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
                                            std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], device = "gpu",
                                            output_layout='FHWC')

    # (frame, channels, height, width) tensor
    video = fn.transpose(video, perm=[0,3,1,2])
    video_target = fn.transpose(video_target, perm=[0, 3, 1, 2])
    return video, video_target



def get_pretrain_video_dataloader(folderpath, folderpath2, batch_size, sequence_length, window_size, first_resize, second_resize , num_worker= 2, device_id = 0):

    pipeline = PreTrainVideoDataPipeline(folderpath, folderpath2, sequence_length, window_size,first_resize, second_resize, batch_size = batch_size,
                                         num_threads = num_worker, device_id = device_id)

    dataloader = DALIGenericIterator(pipeline, output_map=['video', 'video_target'], reader_name="video_replay_reader",
                                     auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)
    steps_per_epoch = pipeline.epoch_size("video_replay_reader") // batch_size
    print("loading video dali dataloader")
    return dataloader, steps_per_epoch

def test_loader():
    # (frame, channels, height, width) tensor
    dali_iter, _ = get_pretrain_video_dataloader(
        "/home/sukai/Documents/Sukai_Project/MontezumaRevenge_human/pretrain_visual_encoder_video_dataset_256",
        "/home/sukai/Documents/Sukai_Project/MontezumaRevenge_human/pretrain_visual_encoder_annotated_video_dataset_256",
        batch_size=32, sequence_length=150, window_size=8, first_resize=256, second_resize=256)


    printflag = False
    for i in range(2):
        tqdm_loader = tqdm(dali_iter)
        for data in tqdm_loader:
            if not printflag:
                printflag = True
                print("image shape: {}".format(data[0]["video"].shape))
                print("image shape: {}".format(data[0]["video_target"].shape))
                print("image device: {}".format(data[0]["video"].device))
                print("Image value max")
                print(torch.max(data[0]['video']))


    # loading video dali dataloader
    # image shape: torch.Size([32, 150, 3, 84, 84])
    # image device: cuda:0
    # Image value max
    # tensor(2.5703, device='cuda:0')

if __name__ == '__main__':
    test_loader()