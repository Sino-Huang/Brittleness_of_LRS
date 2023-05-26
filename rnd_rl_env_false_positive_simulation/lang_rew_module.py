import glob
import os
import pickle

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from backbones import RMSNorm
from models.lang_rew_shaping_event_detection_ver import RewardShapingModule as LangRewEventDet
from models.lang_rew_shaping_binary_classification_ver import RewardShapingModule as LangRewBiCls
from goyal_original.reward_shaping_module import RewardShapingModule as LangRewGoyal


def create_lang_rew_network(cfg):
    # Init Lang Rew Network
    lang_rew_mode = cfg.rl_params.lang_rew_mode
    whether_hard_instruction = cfg.rl_data_files.whether_hard_instruction

    task_difficulty = int(cfg.rl_data_files.rl_task)
    if task_difficulty == 0:
        with open(os.path.join(get_original_cwd(), cfg.rl_data_files.easy_task_saved_state), 'rb') as f:
            task_saved_state = pickle.load(f)

        if lang_rew_mode in [3]:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.easy_task_walkthrough_filepath_goyal)
        elif whether_hard_instruction:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.easy_task_walkthrough_filepath)
        else:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.easy_task_walkthrough_filepath_no_hard)



    elif task_difficulty == 1:
        with open(os.path.join(get_original_cwd(), cfg.rl_data_files.normal_task_saved_state), 'rb') as f:
            task_saved_state = pickle.load(f)

        if lang_rew_mode in [3]:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.normal_task_walkthrough_filepath_goyal)
        elif whether_hard_instruction:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.normal_task_walkthrough_filepath)
        else:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.normal_task_walkthrough_filepath_no_hard)


    elif task_difficulty == 2:
        with open(os.path.join(get_original_cwd(), cfg.rl_data_files.hard_task_saved_state), 'rb') as f:
            task_saved_state = pickle.load(f)

        if lang_rew_mode in [3]:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.hard_task_walkthrough_filepath_goyal)
        elif whether_hard_instruction:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.hard_task_walkthrough_filepath)
        else:
            walkthrough_filepath = os.path.join(get_original_cwd(), cfg.rl_data_files.hard_task_walkthrough_filepath_no_hard)

    else:
        raise NotImplementedError

    with open(walkthrough_filepath, 'rb') as f:
        lang_data_dict = pickle.load(f)
        walkthrough_raw_sentence = []
        walkthrough_embds = []
        data_id_arr = sorted(list(lang_data_dict.keys()))
        for data_id in data_id_arr:
            if lang_rew_mode in [3]:
                walkthrough_raw_sentence.append(lang_data_dict[data_id]['sentence'])
                walkthrough_embds.append(lang_data_dict[data_id]['embedding'])
            else:
                walkthrough_raw_sentence.append(lang_data_dict[data_id]['spell_correction_txt'])
                walkthrough_embds.append(lang_data_dict[data_id]['embedding'])

    walkthrough_embds = np.asarray(walkthrough_embds, dtype=np.float16)

    # walkthrough data arr loading for simulation
    if cfg.rl_params.whether_shorter_chunks:
        walkthrough_data_arr_folderpath = os.path.join(get_original_cwd(), f'rnd_rl_env_false_positive_simulation/annotation_data/task_id_{task_difficulty}/walkthrough_data_shorter_chunks')
    else:
        walkthrough_data_arr_folderpath = os.path.join(get_original_cwd(), f'rnd_rl_env_false_positive_simulation/annotation_data/task_id_{task_difficulty}/walkthrough_data')

    if task_difficulty == 0:
        sent_len = 3
    elif task_difficulty == 1:
        sent_len = 5
    elif task_difficulty == 2:
        sent_len = 5
    else:
        raise NotImplementedError

    walkthrough_data_arr_filespaths = [f'{walkthrough_data_arr_folderpath}/constraints_info_walkthrough_sentence_{i}.pkl' for i in range(1, sent_len + 1)]
    # constraints_info...pkl contains two keys
    # "action_freq" and "avatar_location", one represent the action constraints and the other one control the state constraints.

    walkthrough_data_arr = []
    for p in walkthrough_data_arr_filespaths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
            walkthrough_data_arr.append(d)

    return walkthrough_data_arr, walkthrough_raw_sentence, walkthrough_embds, task_saved_state, task_difficulty



