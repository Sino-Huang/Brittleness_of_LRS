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
from PIL import Image
from einops import rearrange
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

# ```
# dict: {<data_id> : {
#                     "spell_correction_txt": type->str,
#                     "embedding": type-> ndarray(768,),
#                     "phrase_chunking_sentence": type -> lst(str),
#                     "phrase_chunking_tag": type-> lst(str),
#                     'verb_phrase_arr': lst(str)
#                     'noun_phrase_arr': lst(str)
#                     "action_polluted_embedding": type-> lst(ndarray(768,)),
#                     "noun_polluted_embedding": type-> lst(ndarray(768,)),
#                     "both_polluted_embedding": type-> lst(ndarray(768,)),
#                     }
#       }
# ```

class DALIExternalTrainingHardNegativeInputIterator(IterableDataset):
    def __init__(self, cfg, device_id, is_validation):
        super().__init__()

        self.cfg = cfg
        self.batch_size = cfg.lang_rew_shaping_params.batch_size
        self.img_size = cfg.CONSTANT.LW_RESIZE
        self.event_frame_no = cfg.CONSTANT.CLIP_LENGTH // cfg.CONSTANT.PRETRAIN_WINDOW_SIZE
        self.normal_negative_ratio = cfg.lang_rew_shaping_params.normal_negative_ratio
        self.is_validation = is_validation


        self.video_path_arr = None
        self.txt_data_dict = None

        # LOAD DATA
        # load video file path
        self.video_path_arr = glob.glob(
            os.path.join(get_original_cwd(), cfg.data_files.train_video_data_folder_path) + "/*.mp4")
        self.video_dict = dict()
        for videopath in self.video_path_arr:
            videobasename = os.path.basename(videopath)
            filename, _ = videobasename.split('.')
            data_id, _, si, _, ei = filename.split('_')
            data_id, si, ei = int(data_id), int(si), int(ei)
            self.video_dict[data_id] = {
                'filepath': videopath,
                'si': si,
                'ei': ei
            }

        # load instruction data
        with open(os.path.join(get_original_cwd(), cfg.data_files.train_data_dict_hard_negatives_path), 'rb') as f:
            self.txt_data_dict = pickle.load(f)

        # PRE-PROCESSING
        # store data ids as list, more convenient for dali dataloader
        self.ids_arr = []

        size_count = 0
        for key in self.txt_data_dict.keys():
            if self.is_validation:
                if size_count > 0.85 * len(self.txt_data_dict):
                    self.ids_arr.append(key)
            else:
                if size_count <= 0.85 * len(self.txt_data_dict):
                    self.ids_arr.append(key)
            size_count += 1

        # -----------------------------------------------
        # set total data size
        self.total_size = len(self.ids_arr)

        self.ids_arr = self.ids_arr[self.total_size * device_id // cfg.device_info.num_gpus:
                                    self.total_size * (device_id + 1) // cfg.device_info.num_gpus]
        # this is for the internal iteration count, considering parallel shard
        self.n = len(self.ids_arr)

    def __iter__(self):
        self.i = 0 # i is the current index for the iterated data, for internal use
        # shuffle the data
        if not self.is_validation:
            shuffle(self.ids_arr)
        return self

    def __len__(self):
        return self.total_size # consider negative examples

    def __next__(self):
        '''
        we will generate list of numpy array
        50% are positive examples
        25% are verb polluted hard negative examples
        25% are noun polluted hard negative examples
        :return: batchlangs, batchimgs, batchlabels
        '''

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batchlangs = []
        batchimgs = []
        batchlabels = []


        for _ in range(self.batch_size): # consider negative examples
            # we always need video data
            data_id = self.ids_arr[self.i % self.n]

            clipimgs_vec = np.zeros((self.event_frame_no, self.img_size, self.img_size,
                                     3), dtype=np.uint8)  # t: np.ndarray, shape: [len, 84,84,3]
            clip_ind = 0


            video_path = self.video_dict[data_id]['filepath']
            si = self.video_dict[data_id]['si']  # global
            # ei = self.video_dict[data_id]['ei']  # global


            cap = cv2.VideoCapture(video_path)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")
            # Read until video is completed
            frame_count = 0
            while (cap.isOpened()):
                # Capture frame-by-frame
                if frame_count < si:
                    cap.read()
                    frame_count += 1
                else:
                    ret, frame = cap.read()
                    if ret:
                        # Display the resulting frame
                        # resize
                        frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                        # cvt color to rgb
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        clipimgs_vec[clip_ind] = frame
                        clip_ind += 1
                        frame_count += 1
                    else:
                        # means the video ends but the we are not yet satisfied
                        clipimgs_vec[clip_ind] = clipimgs_vec[clip_ind - 1]
                        clip_ind += 1
                        frame_count += 1

                    if clip_ind >= self.event_frame_no:
                        break
            # When everything done, release the video capture object
            cap.release()




            # possible we get neg sample
            neg_sample_rand_val = np.random.rand() # 0 - 1
            if neg_sample_rand_val < 0.5 * self.normal_negative_ratio:
                # both polluted hard negative or normal negative
                label_vec = np.array(0, dtype=np.int64)
                # ------- normal Negative example -------------
                while True:
                    alt_i = random.randint(0, self.n - 1)
                    if alt_i != self.i:
                        break
                alt_data_id = self.ids_arr[alt_i]
                instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(np.float32)  # t: np.ndarray [feature_dim,] TODO, check effectiveness
                # ------- End normal Negative example -------
            elif neg_sample_rand_val < (0.5 * self.normal_negative_ratio + (0.5 - (0.5 * self.normal_negative_ratio))/2.0) and self.txt_data_dict[data_id]['action_polluted_embedding'] is not None:
                # verb polluted hard negative
                label_vec = np.array(0, dtype=np.int64)
                instruction_vec = random.choice(self.txt_data_dict[data_id]['action_polluted_embedding']).astype(
                    np.float32)  # t: np.ndarray [feature_dim,]
            elif neg_sample_rand_val < 0.5 and self.txt_data_dict[data_id]['noun_polluted_embedding'] is not None:
                # noun polluted hard negative
                label_vec = np.array(0, dtype=np.int64)
                instruction_vec = random.choice(self.txt_data_dict[data_id]['noun_polluted_embedding']).astype(
                    np.float32)  # t: np.ndarray [feature_dim,]
            else:
                # positive example
                label_vec = np.array(1, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
                instruction_vec = self.txt_data_dict[data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]


            # append negative data into batch
            batchlangs.append(instruction_vec)
            batchimgs.append(clipimgs_vec)
            batchlabels.append(label_vec)

            self.i += 1

        return batchlangs, batchimgs, batchlabels



class DALIExternalTestingHardNegativeInputIterator(IterableDataset):
    def __init__(self, cfg, device_id):
        super().__init__()

        self.cfg = cfg
        self.batch_size = 1
        self.img_size = cfg.CONSTANT.LW_RESIZE
        self.total_size_ratio = 3

        self.video_path_arr = None
        self.txt_data_dict = None

        # LOAD DATA
        # load video file path
        self.video_path_arr = glob.glob(
            os.path.join(get_original_cwd(), cfg.data_files.test_video_data_folder_path) + "/*.mp4")
        self.video_dict = dict()
        for videopath in self.video_path_arr:
            videobasename = os.path.basename(videopath)
            filename, _ = videobasename.split('.')
            data_id, _, si, _, ei = filename.split('_')
            data_id, si, ei = int(data_id), int(si), int(ei)
            self.video_dict[data_id] = {
                'filepath': videopath,
                'si': si,
                'ei': ei
            }

        # load instruction data
        with open(os.path.join(get_original_cwd(), cfg.data_files.test_data_dict_path), 'rb') as f:
            self.txt_data_dict = pickle.load(f)

        # PRE-PROCESSING
        # store data ids as list, more convenient for dali dataloader
        self.ids_arr = []

        for key in self.txt_data_dict.keys():
            self.ids_arr.append(key)

        # -----------------------------------------------
        # set total data size
        self.total_size = len(self.ids_arr)

        self.ids_arr = self.ids_arr[self.total_size * device_id // cfg.device_info.num_gpus:
                                    self.total_size * (device_id + 1) // cfg.device_info.num_gpus]
        # this is for the internal iteration count, considering parallel shard
        self.n = len(self.ids_arr)

    def __iter__(self):
        self.i = 0 # i is the current index for the iterated data, for internal use
        return self

    def __len__(self):
        return self.total_size * self.total_size_ratio # consider the negatives

    def __next__(self):
        '''
        we will generate list of numpy array
        50% are positive examples
        25% are verb polluted hard negative examples
        25% are noun polluted hard negative examples
        :return: batchlangs, batchimgs, batchlabels
        '''

        if self.i >= self.n * self.total_size_ratio: # consider negatives
            self.__iter__()
            raise StopIteration

        batchlangs = []
        batchimgs = []
        batchlabels = []


        for _ in range(self.batch_size):
            # we always need video data
            data_id = self.ids_arr[self.i % self.n]

            video_path = self.video_dict[data_id]['filepath']

            si = self.video_dict[data_id]['si']  # global
            ei = self.video_dict[data_id]['ei']  # global
            clipimgs_vec = np.zeros((ei - si, self.img_size, self.img_size,
                                        3), dtype=np.uint8)  # t: np.ndarray, shape: [len, 84,84,3]
            clip_ind = 0

            cap = cv2.VideoCapture(video_path)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")
            # Read until video is completed
            frame_count = 0
            while (cap.isOpened()):
                # Capture frame-by-frame
                if frame_count < si:
                    cap.read()
                    frame_count += 1
                else:
                    ret, frame = cap.read()
                    if ret:
                        # Display the resulting frame
                        # resize
                        frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                        # cvt color to rgb
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        clipimgs_vec[clip_ind] = frame
                        clip_ind += 1
                        frame_count += 1
                    else:
                        # means the video ends but the we are not yet satisfied
                        clipimgs_vec[clip_ind] = clipimgs_vec[clip_ind - 1]
                        clip_ind += 1
                        frame_count += 1

                    if frame_count >= ei:
                        break
            # When everything done, release the video capture object
            cap.release()

            if self.i < self.n: # this means the positive example
                label_vec = np.array(1, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
                instruction_vec = self.txt_data_dict[data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]
            elif self.i < self.n * 2: # negative examples
                alt_data_id = self.ids_arr[ (self.i + 1 )% self.n]
                label_vec = np.array(0, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
                instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]

            elif self.i < self.n * 3: # negative examples
                alt_data_id = self.ids_arr[ (self.i + 2 )% self.n]
                label_vec = np.array(0, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
                instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]

            # elif self.i < self.n * 4: # negative examples
            #     alt_data_id = self.ids_arr[ (self.i + 3 )% self.n]
            #     label_vec = np.array(0, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
            #     instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(
            #         np.float32)  # t: np.ndarray [feature_dim,]

            # append data into batch
            batchlangs.append(instruction_vec)
            batchimgs.append(clipimgs_vec)
            batchlabels.append(label_vec)
            self.i += 1

        return batchlangs, batchimgs, batchlabels


@pipeline_def
def ExternalSourcePipelineWithAugmentation(cfg, external_data):
    batchlangs, batchimgs, batchlabels = fn.external_source(source=external_data,
                                                                          num_outputs=3,
                                                                          device='cpu',
                                                                          batch=True,
                                                                          dtype=[
                                                                              types.FLOAT,
                                                                              types.UINT8,
                                                                              types.INT64
                                                                          ])
    if cfg.device_info.device != "cpu":
        batchimgs = batchimgs.gpu()
        batchlangs = batchlangs.gpu()
        batchlabels = batchlabels.gpu()


    batchimgs = fn.transpose(batchimgs, perm=[0, 3, 1, 2], output_layout=types.NFCHW, device="gpu" if cfg.device_info.device != "cpu" else "cpu")


    batchimgs = fn.crop_mirror_normalize(batchimgs, dtype=types.FLOAT, mean=eval(cfg.CONSTANT.IMG_MEAN),
                                         std=eval(cfg.CONSTANT.IMG_STD), output_layout=types.NFCHW,
                                         device="gpu" if cfg.device_info.device != "cpu" else "cpu")


    return batchlangs, batchimgs, batchlabels


def get_hard_negative_dataloader(cfg, is_train, is_validation):
    batch_size = 1
    if is_train:
        external_data = DALIExternalTrainingHardNegativeInputIterator(cfg, cfg.device_info.device_id, is_validation)
        batch_size = cfg.lang_rew_shaping_params.batch_size

    else:
        external_data = DALIExternalTestingHardNegativeInputIterator(cfg, cfg.device_info.device_id)

    data_pipeline = ExternalSourcePipelineWithAugmentation(cfg=cfg, external_data=external_data,
                                                           batch_size = batch_size,
                                                           num_threads = cfg.device_info.num_worker,
                                                           device_id = cfg.device_info.device_id)

    if batch_size == 1:
        data_loader_size = len(external_data)
    else:
        data_loader_size = (batch_size - len(external_data) % batch_size) + len(external_data)

    data_loader = DALIGenericIterator(data_pipeline, output_map=['lang',
                                                                 'video',
                                                                 'label',
                                                                 ],
                                      size=data_loader_size,
                                      auto_reset=True)

    steps_per_epoch = math.ceil(len(external_data) / batch_size)
    return data_loader, steps_per_epoch


@hydra.main(version_base=None, config_path="../config", config_name="lang_rew_module")
def main(cfg: DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    train_loader, _ = get_hard_negative_dataloader(cfg, True, False)
    valid_loader, _ = get_hard_negative_dataloader(cfg, True, True)
    test_loader, _ = get_hard_negative_dataloader(cfg, False, True)

    trainflag = False
    testflag = False

    for i in range(2):
        tqdm_train_loader = tqdm(train_loader)
        for data in tqdm_train_loader:
            if not trainflag:
                trainflag = True
                print(data[0]['lang'].shape)
                print(data[0]['video'].shape)
                print(data[0]['label'].shape)
                # a = data[0]['video'][2][10].detach().cpu().numpy()
                # im = Image.fromarray(a)
                # im.save(os.path.join( get_original_cwd(),"debug_one.jpeg"))
                # b = data[0]['video'][2][11].detach().cpu().numpy()
                # im = Image.fromarray(b)
                # im.save(os.path.join(get_original_cwd(), "debug_two.jpeg"))

                # print("Image value")
                # print(data[0]['image'][0][0])

        tqdm_test_loader = tqdm(valid_loader)
        for data in tqdm_test_loader:
            if not testflag:
                testflag = True
                print(data[0]['lang'].shape)
                print(data[0]['video'].shape)
                print(data[0]['label'].shape)
                # print("Image value")
                # print(data[0]['image'][0][0])
    # lang shape: torch.Size([32, 768])
    # image shape: torch.Size([32, 18, 3, 84, 84])   5.34it/s
    # image are normalised by ImageNet mean and std
    # label shape torch.Size([32])


if __name__ == '__main__':
    main()
    # temp()
