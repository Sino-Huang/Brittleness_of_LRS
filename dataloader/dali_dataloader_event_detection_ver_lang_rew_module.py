import ast
import io
import math
import pickle
from collections import OrderedDict

import cv2
import numpy as np
import torch
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
import glob

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

# after testing we find that given limited training data, we cannot train this event det module quite well.
# therefore we can simplify the training task by concatenating the two dis-consecutive clip
# our training length can be from 1 * 18 to 4 * 18
# for 1 and 2 it is possible to get negative examples
class DALIExternalTrainingVideoSentEmbInputIterator(IterableDataset):
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
        self.video_path_arr = glob.glob(os.path.join(get_original_cwd(), cfg.data_files.train_video_data_folder_path) + "/*.mp4")
        self.video_dict = dict()
        for videopath in self.video_path_arr:
            videobasename = os.path.basename(videopath)
            filename, _ = videobasename.split('.')
            data_id, _ ,si, _, ei = filename.split('_')
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
                                    self.total_size * (device_id+1) // cfg.device_info.num_gpus]
        # this is for the internal iteration count, considering parallel shard
        self.n = len(self.ids_arr)

    def __iter__(self):
        self.i = 0 # i is the current index for the iterated data, for internal use
        # shuffle the data
        if not self.is_validation:
            shuffle(self.ids_arr)
        return self

    def __len__(self):
        return self.total_size

    def __next__(self):
        '''
        we will generate list of numpy array
        :return: batchlangs, batchimgs, batchlabels
        '''

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        batchlangs = []
        batchimgs = []
        batchlabels = [] # starting point, duration (all normalised), for negative example, we do not consider the starting point and duration

        # determine video len for current batch min 2*18, max 3*18
        # video_len = random.choice([1,2,3,4])
        video_len = 2


        for _ in range(self.batch_size):
            data_id = self.ids_arr[self.i % self.n]
            # instruction
            instruction_vec = np.asarray(self.txt_data_dict[data_id]['embedding'], dtype=np.float32)
            neg_sample_rand_val = np.random.rand()  # 0 - 1

            if video_len <= 2 and neg_sample_rand_val < 0.5 * self.normal_negative_ratio: # this means we may get negative examples
                alt_data_id_arr = []
                while len(alt_data_id_arr) < video_len:
                    while True:
                        alt_i = random.randint(0, self.n - 1)
                        if alt_i != self.i % self.n:
                            break
                    alt_data_id = self.ids_arr[alt_i]
                    alt_data_id_arr.append(alt_data_id)
                frames = []
                for alt_data_id in alt_data_id_arr:
                    video_path = self.video_dict[alt_data_id]['filepath']
                    si = self.video_dict[alt_data_id]['si']  # global
                    ei = si + self.event_frame_no  # global
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
                                frame = cv2.resize(frame, (self.img_size, self.img_size),
                                                   interpolation=cv2.INTER_AREA)
                                # cvt color to rgb
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame)
                                frame_count += 1
                            else:
                                # means the video ends but the we are not yet satisfied
                                frames.append(frames[-1])
                                frame_count += 1

                            if frame_count >= ei:
                                break
                    # When everything done, release the video capture object
                    cap.release()
                clipimgs_vec = np.asarray(frames, dtype=np.uint8)
                label_vec = np.array([np.NINF, np.NINF], dtype=np.float32)

            else:
                # concatenate x+1 clips together
                all_data_id_arr = []
                while len(all_data_id_arr) < video_len + 1:
                    while True:
                        alt_i = random.randint(0, self.n - 1)
                        if alt_i != self.i % self.n:
                            break
                    alt_data_id = self.ids_arr[alt_i]
                    all_data_id_arr.append(alt_data_id)
                all_data_id_arr[len(all_data_id_arr)// 2] = data_id
                pos_example_location = all_data_id_arr.index(data_id)

                frames = []
                for all_data_id in all_data_id_arr:
                    video_path = self.video_dict[all_data_id]['filepath']
                    si = self.video_dict[all_data_id]['si']  # global
                    ei = si + self.event_frame_no  # global
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
                                frame = cv2.resize(frame, (self.img_size, self.img_size),
                                                   interpolation=cv2.INTER_AREA)
                                # cvt color to rgb
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame)
                                frame_count += 1
                            else:
                                # means the video ends but the we are not yet satisfied
                                frames.append(frames[-1])
                                frame_count += 1

                            if frame_count >= ei:
                                break
                    # When everything done, release the video capture object
                    cap.release()

                # now consider the offset
                offset = random.randint(0, self.event_frame_no - 1)
                frames = frames[offset: offset + video_len * self.event_frame_no]
                clipimgs_vec = np.asarray(frames, dtype=np.uint8)

                if pos_example_location != 0: # pos_example_location in [1,2,3,4...]
                    event_starting_ind_norm = (pos_example_location * self.event_frame_no - offset) / (video_len * self.event_frame_no)
                else:
                    event_starting_ind_norm = 0.0

                if pos_example_location == 0:
                    event_duration_norm = (self.event_frame_no - offset) / (video_len * self.event_frame_no)
                elif pos_example_location == video_len:
                    event_duration_norm = offset / (video_len * self.event_frame_no)
                else:
                    event_duration_norm = self.event_frame_no / (video_len * self.event_frame_no)

                label_vec = np.array([event_starting_ind_norm, event_duration_norm], dtype=np.float32)

                if neg_sample_rand_val < 0.5: # hard negative
                    neg_sample_rand_temp = np.random.rand()
                    if neg_sample_rand_temp < 0.5 and \
                            self.txt_data_dict[data_id]['action_polluted_embedding'] is not None:
                        # verb polluted hard negative
                        label_vec = np.array([np.NINF, np.NINF], dtype=np.float32)
                        instruction_vec = random.choice(
                            self.txt_data_dict[data_id]['action_polluted_embedding']).astype(
                            np.float32)  # t: np.ndarray [feature_dim,]
                    elif neg_sample_rand_temp < 1.0 and self.txt_data_dict[data_id][
                        'noun_polluted_embedding'] is not None:
                        # noun polluted hard negative
                        label_vec = np.array([np.NINF, np.NINF], dtype=np.float32)
                        instruction_vec = random.choice(self.txt_data_dict[data_id]['noun_polluted_embedding']).astype(
                            np.float32)  # t: np.ndarray [feature_dim,]


            # append data into batch
            batchlangs.append(instruction_vec)
            batchimgs.append(clipimgs_vec)
            batchlabels.append(label_vec)
            self.i += 1

        return batchlangs, batchimgs, batchlabels


class DALIExternalTestingVideoSentEmbInputIterator(IterableDataset):
    def __init__(self, cfg, device_id):
        super().__init__()

        self.cfg = cfg
        self.batch_size = 1
        self.img_size = cfg.CONSTANT.LW_RESIZE
        self.event_frame_no = cfg.CONSTANT.CLIP_LENGTH // cfg.CONSTANT.PRETRAIN_WINDOW_SIZE
        self.total_size_ratio = 2


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
        self.i = 0  # i is the current index for the iterated data, for internal use
        return self

    def __len__(self):
        return self.total_size * self.total_size_ratio # consider the negatives

    def __next__(self):
        '''
        we will generate list of numpy array
        :return: batchlangs, batchimgs, batchactions, batchlabels
        '''

        if self.i >= self.n * self.total_size_ratio: # consider negatives
            self.__iter__()
            raise StopIteration

        batchlangs = []
        batchimgs = []
        batchlabels = []  # starting point, duration (all normalised)

        for _ in range(self.batch_size):
            data_id = self.ids_arr[self.i % self.n]

            video_path = self.video_dict[data_id]['filepath']
            si = self.video_dict[data_id]['si']  # global
            ei = self.video_dict[data_id]['ei']  # global


            cap = cv2.VideoCapture(video_path)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")
            # Read until video is completed
            frame_arr = []
            while (cap.isOpened()):
                # Capture frame-by-frame

                ret, frame = cap.read()
                if ret:
                    # Display the resulting frame
                    # resize
                    frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                    # cvt color to rgb
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_arr.append(frame)
                else:
                    break
            # When everything done, release the video capture object
            cap.release()
            video_len = len(frame_arr)
            clipimgs_vec = np.asarray(frame_arr, dtype=np.uint8)

            # determine event duration norm, due to training data limitation, we only have fixed event duration
            event_duration_norm = (ei - si) / video_len
            event_starting_ind_norm = si / video_len

            if self.i < self.n:  # this means the positive example
                # instruction
                instruction_vec = np.asarray(self.txt_data_dict[data_id]['embedding'], dtype=np.float32)
                # outputlabel
                label_vec = np.array([event_starting_ind_norm, event_duration_norm], dtype=np.float32)
            elif self.i < self.n * 2: # negative examples
                alt_data_id = self.ids_arr[ (self.i + 2 )% self.n]
                label_vec = np.array([np.NINF, np.NINF], dtype=np.float32)  # NINF means not matched (neg)
                instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]

            # elif self.i < self.n * 3: # negative examples
            #     alt_data_id = self.ids_arr[ (self.i + 3 )% self.n]
            #     label_vec = np.array([np.NINF, np.NINF], dtype=np.float32)  # NINF means not matched (neg)
            #     instruction_vec = self.txt_data_dict[alt_data_id]['embedding'].astype(
            #         np.float32)  # t: np.ndarray [feature_dim,]



            # append data into batch
            batchlangs.append(instruction_vec)
            batchimgs.append(clipimgs_vec)
            batchlabels.append(label_vec)
            self.i += 1

        return batchlangs, batchimgs, batchlabels



@pipeline_def
def ExternalSourcePipeline(cfg, external_data):
    batchlangs, batchimgs, batchlabels = fn.external_source(source=external_data,
                                                                          num_outputs=3,
                                                                          device='cpu',
                                                                          batch=True,
                                                                          dtype=[
                                                                              types.FLOAT,
                                                                              types.UINT8,
                                                                              types.FLOAT
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

def get_loader_sent_emb(cfg, is_train, is_validation):
    batch_size = 1
    if is_train:
        external_data = DALIExternalTrainingVideoSentEmbInputIterator(cfg, cfg.device_info.device_id, is_validation)
        batch_size = cfg.lang_rew_shaping_params.batch_size
    else:
        external_data = DALIExternalTestingVideoSentEmbInputIterator(cfg, cfg.device_info.device_id)

    data_pipeline = ExternalSourcePipeline(cfg=cfg, external_data=external_data, batch_size = batch_size,
                                         num_threads = cfg.device_info.num_worker, device_id = cfg.device_info.device_id)

    if batch_size == 1:
        data_loader_size = len(external_data)
    else:
        data_loader_size = (batch_size - len(external_data) % batch_size) + len(external_data)

    data_loader = DALIGenericIterator(data_pipeline, output_map=['lang',
                                                                 'video',
                                                                 'label'
                                                                 ], size=data_loader_size,
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

    train_loader, _ = get_loader_sent_emb(cfg, True, False)
    valid_loader, _ = get_loader_sent_emb(cfg, True, True)
    test_loader, _ = get_loader_sent_emb(cfg, False, True)

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
    # lang shape: torch.Size([32, 768]) for sentence emb
    # image shape: torch.Size([32, 38, 3, 84, 84])
    # image are normalised by ImageNet mean and std
    # label shape: torch.Size([32, 2])

if __name__ == '__main__':
    main()