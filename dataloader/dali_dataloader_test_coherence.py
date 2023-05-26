import ast
import glob
import io
import logging
import math
import pickle
import sys
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
from dataloader import DALIExternalTestingVideoSentEmbInputIterator
import torch.nn as nn
from models.lang_rew_shaping_event_detection_ver.optimizer import get_optims_and_scheduler
from rl_env.atari_wrapper import MR_ATARI_ACTION

log = logging.getLogger(__name__)



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

class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, factor=4, dropout_rate=0.2, norm_layer=nn.Identity):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor
        self.middle_dim = int(self.factor * self.input_dim)
        self.block = nn.Sequential(
            nn.Linear(self.input_dim, self.middle_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            norm_layer(self.middle_dim),
            nn.Linear(self.middle_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DALIExternalTrainingHardNegativeInputIterator(IterableDataset):
    def __init__(self, cfg, device_id):
        super().__init__()

        self.cfg = cfg
        self.batch_size = cfg.lang_rew_shaping_params.batch_size
        self.instruction_emb_dict = None

        # LOAD DATA
        # load instruction data
        with open(os.path.join(get_original_cwd(), cfg.data_files.train_data_instruction_dict_path), 'rb') as f:
            self.instruction_emb_dict = pickle.load(f)

        # PRE-PROCESSING
        # store data ids as list, more convenient for dali dataloader
        self.ids_arr = []

        for key in self.instruction_emb_dict.keys():
            self.ids_arr.append(key)

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
        shuffle(self.ids_arr)
        return self

    def __len__(self):
        return self.total_size

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
        batchlabels = []

        for _ in range(self.batch_size):
            # we always need
            data_id = self.ids_arr[self.i % self.n]

            # possible we get neg sample
            # action_polluted_embedding, noun_polluted_embedding
            neg_sample_rand_val = np.random.rand() # 0 - 0.25 verb polluted hard negative, 0.25-0.5 noun polluted
            if neg_sample_rand_val < 0.25 and self.instruction_emb_dict[data_id]['noun_polluted_embedding'] is not None:
                # noun polluted
                label_vec = np.array(0, dtype=np.int64)
                instruction_vec = random.choice(self.instruction_emb_dict[data_id]['noun_polluted_embedding']).astype(
                    np.float32)
            elif neg_sample_rand_val < 0.5 and self.instruction_emb_dict[data_id]['action_polluted_embedding'] is not None:
                # action polluted
                label_vec = np.array(0, dtype=np.int64)
                instruction_vec = random.choice(self.instruction_emb_dict[data_id]['action_polluted_embedding']).astype(
                    np.float32)


            else:
                # normal example
                label_vec = np.array(1, dtype=np.int64)  # 1 means matched, 0 means not matched (neg)
                instruction_vec = self.instruction_emb_dict[data_id]['embedding'].astype(
                    np.float32)  # t: np.ndarray [feature_dim,]


            # append data into batch
            batchlangs.append(instruction_vec)
            batchlabels.append(label_vec)
            self.i += 1

        return batchlangs, batchlabels


@pipeline_def
def ExternalSourcePipelineWithAugmentation(cfg, external_data):
    batchlangs, batchlabels = fn.external_source(source=external_data,
                                                                          num_outputs=2,
                                                                          device='cpu',
                                                                          batch=True,
                                                                          dtype=[
                                                                              types.FLOAT,
                                                                              types.INT64
                                                                          ])
    if cfg.device_info.device != "cpu":
        batchlangs = batchlangs.gpu()
        batchlabels = batchlabels.gpu()


    return batchlangs, batchlabels


def get_hard_negative_dataloader(cfg, is_train):
    if is_train:
        external_data = DALIExternalTrainingHardNegativeInputIterator(cfg, cfg.device_info.device_id)
    else:
        external_data = DALIExternalTestingVideoSentEmbInputIterator(cfg, cfg.device_info.device_id)
    data_pipeline = ExternalSourcePipelineWithAugmentation(cfg=cfg, external_data=external_data,
                                                           batch_size = cfg.lang_rew_shaping_params.batch_size,
                                                           num_threads = cfg.device_info.num_worker,
                                                           device_id = cfg.device_info.device_id)
    data_loader = DALIGenericIterator(data_pipeline, output_map=['lang',
                                                                 'label',
                                                                 ],
                                      size=(cfg.lang_rew_shaping_params.batch_size - len(external_data)
                                            % cfg.lang_rew_shaping_params.batch_size) + len(external_data),
                                      auto_reset=True)

    steps_per_epoch = math.ceil(len(external_data) / cfg.lang_rew_shaping_params.batch_size)
    return data_loader, steps_per_epoch


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    # classification_loss_function = torch.nn.BCEWithLogitsLoss()                                        # CHANGE
    classification_loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device, non_blocking=True)
    accu_num = torch.zeros(1).to(device, non_blocking=True)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for data in data_loader:
        langs = data[0]['lang']
        langs = langs.to(device, non_blocking= True)

        labels = data[0]['label']
        labels = labels.to(device, non_blocking= True)

        # handle data
        sample_num += labels.shape[0]

        pred = model(langs)
        # label shape: torch.Size([32])
        # pred_classes = torch.round(pred).long() # this output the ind -> 1 is normal, 0 is polluted # CHANGE
        pred_classes = torch.max(pred, dim=1)[1] # this output the max indices

        # accu_num += torch.eq(pred_classes, labels.unsqueeze(-1)).sum()                                  # CHANGE
        accu_num += torch.eq(pred_classes, labels).sum()


        # loss = classification_loss_function(pred, labels.unsqueeze(-1).float())                         # CHANGE
        loss = classification_loss_function(pred, labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / sample_num,
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
    return accu_loss.item() / sample_num, accu_num.item() / sample_num



@hydra.main(version_base=None, config_path="../config", config_name="lang_rew_module_sentence_emb_hard_negative_version")
def test_coherence(cfg: DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    train_dataloader, steps_per_epoch = get_hard_negative_dataloader(cfg, True)

    model = MLPBlock(768, 2).to(cfg.device_info.device)


    # Optimizer and lr scheduler
    optimizer, lr_scheduler = get_optims_and_scheduler(model, cfg, steps_per_epoch)
    for epoch in range(cfg.lang_rew_shaping_params.n_epochs):
        # train
        loss_train, acc_train = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=cfg.device_info.device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                )

        log.info('Epoch: {:d} \t TL: {:5f} \t TA: {:.2%}'
                 .format(epoch, loss_train, acc_train))




@hydra.main(version_base=None, config_path="../config", config_name="lang_rew_module_sentence_emb_hard_negative_version")
def main(cfg: DictConfig):
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    train_loader, _ = get_hard_negative_dataloader(cfg, True)
    test_loader, _ = get_hard_negative_dataloader(cfg, False)

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

        tqdm_test_loader = tqdm(test_loader)
        for data in tqdm_test_loader:
            if not testflag:
                testflag = True
                print(data[0]['lang'].shape)
                print(data[0]['video'].shape)
                print(data[0]['label'].shape)
                # print("Image value")
                # print(data[0]['image'][0][0])
    # lang shape: torch.Size([32, 768])
    # image shape: torch.Size([32, 14, 3, 84, 84])   5.34it/s
    # image are normalised by ImageNet mean and std
    # action shape: torch.Size([128, 8])


if __name__ == '__main__':
    test_coherence()
    # temp()
