import ast
import glob
import io
import math
import pickle
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from hydra.utils import get_original_cwd, to_absolute_path
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.utils.data import IterableDataset
from tqdm import tqdm
from random import shuffle
import random
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torchvision.transforms as T

from rnd_rl_env.envs import MR_ATARI_ACTION


def fromfile(file, dtype, count, *args, **kwargs):
    """Read from any file-like object into a numpy array
       im = io.BytesIO(f.read())
       m = fromfile(im, dtype=np.uint8, count=im.__sizeof__())"""

    itemsize = np.dtype(dtype).itemsize
    buffer = np.zeros(count * itemsize, np.uint8)
    bytes_read = -1
    offset = 0
    while bytes_read != 0:
        bytes_read = file.readinto(buffer[offset:])
        offset += bytes_read
    rounded_bytes = (offset // itemsize) * itemsize
    buffer = buffer[:rounded_bytes]
    buffer.dtype = dtype
    return buffer


class DALIExternalTrainingVideoActionInputIterator(IterableDataset):
    def __init__(self, cfg, device_id, is_train):
        super().__init__()

        self.cfg = cfg
        self.device_id = device_id
        self.is_train = is_train
        self.batch_size = cfg.visual_action_predictor_params.batch_size if is_train else 1
        self.CLIP_LENGTH = cfg.CONSTANT.CLIP_LENGTH
        self.window_size = cfg.CONSTANT.PRETRAIN_WINDOW_SIZE
        self.img_size = cfg.CONSTANT.LW_RESIZE
        self.IMG_MEAN = np.array([0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0], dtype=np.float32)
        self.IMG_STD = np.array([0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], dtype=np.float32)
        self.IMG_MEAN = repeat(self.IMG_MEAN, 'd -> w h d', w=self.img_size, h=self.img_size)
        self.IMG_STD = repeat(self.IMG_STD, 'd -> w h d', w=self.img_size, h=self.img_size)
        if is_train:
            self.image_training_folder_path = os.path.join(get_original_cwd(),
                                                       cfg.data_files.pretrain_image_jpeg_training_folder)
        else:
            self.image_training_folder_path = os.path.join(get_original_cwd(),
                                                           cfg.data_files.pretrain_image_jpeg_testing_folder)
        self.action_folder_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_action_folder)

        self.image_episode_folder_list = glob.glob(self.image_training_folder_path + "/**/")

        # determine total data size one batch is (relative offset, normalised action) pair
        self.total_size = 0
        self.img_path_action_pair_arr = []
        # this is for the internal iteration count, considering parallel shard
        self.n = 0
        self.setup_total_size_and_batch_loop()

    def setup_total_size_and_batch_loop(self):
        self.img_path_action_pair_arr = []  # ((img_t-1, img_t), normalised_action)
        for img_folder_path in self.image_episode_folder_list:
            img_shrink_path_pair_list = self.setup_episode_training_data(img_folder_path)
            self.img_path_action_pair_arr.extend(img_shrink_path_pair_list)

        self.total_size = len(self.img_path_action_pair_arr)
        self.img_path_action_pair_arr = self.img_path_action_pair_arr[
                                        self.total_size * self.device_id // self.cfg.device_info.num_gpus:
                                        self.total_size * (self.device_id + 1) // self.cfg.device_info.num_gpus]
        self.n = len(self.img_path_action_pair_arr)

    def setup_episode_training_data(self, episode_folder_path):
        # get episode name
        episode_name = episode_folder_path.split('/')[-2]  # type: str

        # loop episode folder
        img_pathlist = glob.glob(episode_folder_path + '/*.jpeg')
        # img_int_name_path_arr= [int(x[:-5].split('/')[-1]) for x in img_pathlist]
        # img_int_name_path_arr.sort() # ascending sort

        # get action info
        traj_info_path = os.path.join(self.action_folder_path, episode_name + '.txt')
        traj_df = pd.read_csv(traj_info_path, skiprows=1, )
        action_df = traj_df[' action']

        # set random offset
        offset = random.randint(0, self.window_size - 1)
        img_shrink_path_pair_list = []  # ((img_t-1, img_t), normalised_action)
        # episode shrink by windowsize
        prev_ind = None
        skip_first_flag = True
        for ind in range(offset, len(img_pathlist), self.window_size):
            if skip_first_flag:
                skip_first_flag = False
                prev_ind = ind
                continue

            img_pair = (f"{episode_name}/{prev_ind}", f"{episode_name}/{ind}")  # eg: ("931/12", "931/28") (previous, after)

            action_seq = action_df.iloc[prev_ind:ind].tolist()
            action_frequncy_vector = np.zeros((len(MR_ATARI_ACTION),), dtype=np.float32)
            for i in action_seq:
                if i == 11 or i == 14 or i == 16:
                    action_frequncy_vector[6] += 1
                elif i == 12 or i == 15 or i == 17:
                    action_frequncy_vector[7] += 1
                elif i == 6 or i == 8: # UPRIGHT = RIGHT
                    action_frequncy_vector[3] += 1
                elif i == 7 or i == 9: # UPLEFT = LEFT
                    action_frequncy_vector[4] += 1
                elif i == 10 or 13:
                    action_frequncy_vector[1] += 1
                else:
                    action_frequncy_vector[i] += 1

            # normalise
            action_frequncy_vector /= np.sum(action_frequncy_vector)
            # append data to the list
            img_shrink_path_pair_list.append((img_pair, action_frequncy_vector))  # input and label all together

            # end, update prev_ind
            prev_ind = ind
        # pairup imgt-1 and imgt

        return img_shrink_path_pair_list

    def __iter__(self):
        self.i = 0  # i is the current index for the iterated data, for internal use
        # shuffle the data
        if self.is_train:
            # self.setup_total_size_and_batch_loop() # reselect the data
            shuffle(self.img_path_action_pair_arr)
        return self

    def __len__(self):
        return self.total_size

    def __next__(self):
        '''
        we will generate list of numpy array
        :return: batchimgs, batchactions
        '''

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batchimgs = [] # shape [batch, W, H]
        batchaction = [] # shape [batch, ACTION_LEN]

        for _ in range(self.batch_size):
            img_pair, action_frequncy_vector = self.img_path_action_pair_arr[self.i % self.n]

            clip_img_arr = [] # previous, and current frame
            for img_ind, img_name in enumerate(img_pair): # example: ("931/12", "931/28")
                img_path = os.path.join(get_original_cwd(), self.image_training_folder_path, img_name+".jpeg" )
                with open(img_path, 'rb') as f:
                    im = io.BytesIO(f.read())
                    im = fromfile(im, dtype=np.uint8, count=im.__sizeof__())
                    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                    im = cv2.resize(im, (self.cfg.CONSTANT.LW_RESIZE, self.cfg.CONSTANT.LW_RESIZE), interpolation=cv2.INTER_AREA)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    # they will have -mean and /std
                    im = (im-self.IMG_MEAN) / self.IMG_STD

                    # other process
                    im = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2] # to greyscale
                    im = torch.asarray(im, dtype=torch.float32)
                    im = rearrange(im, 'w h -> 1 w h')
                    # shape : -> [84, 84]

                    clip_img_arr.append(im)

            # append data into batch
            batchimgs.append(clip_img_arr[-1] - clip_img_arr[0])
            batchaction.append(action_frequncy_vector)
            self.i += 1

        return batchimgs, batchaction




@pipeline_def
def ExternalSourcePipeline(cfg, external_data):
    batchimgs, batchaction = fn.external_source(source=external_data,
                                                            num_outputs=2,
                                                            device='cpu',
                                                            batch=True,
                                                            dtype=[
                                                                types.FLOAT,
                                                                types.FLOAT,
                                                            ])
    if cfg.device_info.device != "cpu":
        batchimgs = batchimgs.gpu()
        batchaction = batchaction.gpu()


    return batchimgs, batchaction


def get_pretrain_video_action_dataloader(cfg, is_train):
    batch_size = cfg.visual_action_predictor_params.batch_size if is_train else 1

    external_data = DALIExternalTrainingVideoActionInputIterator(cfg, cfg.device_info.device_id, is_train)
    data_pipeline = ExternalSourcePipeline(cfg=cfg, external_data=external_data,
                                           batch_size=batch_size,
                                           num_threads=cfg.device_info.num_worker, device_id=cfg.device_info.device_id)

    if batch_size == 1:
        data_loader_size = len(external_data)
    else:
        data_loader_size = (batch_size - len(external_data) % batch_size) + len(external_data)

    data_loader = DALIGenericIterator(data_pipeline, output_map=['img_diff',
                                                                 'action',
                                                                 ],
                                      size=data_loader_size,
                                      auto_reset=True)

    steps_per_epoch = math.ceil(len(external_data) / data_loader_size)
    return data_loader, steps_per_epoch


@hydra.main(version_base=None, config_path="../config", config_name="video_action_pretrain")
def main(cfg: DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    train_loader, _ = get_pretrain_video_action_dataloader(cfg, True)
    test_loader, _ = get_pretrain_video_action_dataloader(cfg, False)

    trainflag = False
    testflag = False

    for i in range(2):
        tqdm_train_loader = tqdm(train_loader)
        for data in tqdm_train_loader:
            if not trainflag:
                trainflag = True
                print(data[0]['img_diff'].shape)
                print(data[0]['action'].shape)
                # print("Image value")
                # print(data[0]['image'][0][0])

        tqdm_test_loader = tqdm(test_loader)
        for data in tqdm_test_loader:
            if not testflag:
                testflag = True
                print(data[0]['img_diff'].shape)
                print(data[0]['action'].shape)
                # print("Image value")
                # print(data[0]['image'][0][0])
    # image shape: torch.Size([128, 1, 84, 84])  21.43it/s
    # image are normalised by ImageNet mean and std
    # action shape: torch.Size([128, 8])


@hydra.main(version_base=None, config_path="../config", config_name="video_action_pretrain")
def temp(cfg: DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    folder_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_image_jpeg_training_folder)
    action_folder_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_action_folder)
    a = glob.glob(folder_path + "/**/")
    txt = glob.glob(action_folder_path + "/*.txt")
    b = 1


if __name__ == '__main__':
    main()
    # temp()
