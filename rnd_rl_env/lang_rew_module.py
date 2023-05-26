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
    want_lang_rew = cfg.rl_params.want_lang_rew
    if want_lang_rew:
        if lang_rew_mode == 1: # binary cls
            lang_network_dictpath = os.path.join(get_original_cwd(), cfg.rl_data_files.lang_rew_module_dictpath_binary_cls)
            langrewclass = LangRewBiCls
            is_goyal = False
            is_binary_cls = True

        elif lang_rew_mode == 2: # 2. binary cls hard neg
            lang_network_dictpath = os.path.join(get_original_cwd(), cfg.rl_data_files.lang_rew_module_dictpath_binary_cls_hard_negative)
            langrewclass = LangRewBiCls
            is_goyal = False
            is_binary_cls = True


        elif lang_rew_mode == 3: # goyal
            lang_network_dictpath = os.path.join(get_original_cwd(), cfg.rl_data_files.lang_rew_module_dictpath_goyal)
            langrewclass = LangRewGoyal
            is_goyal = True
            is_binary_cls = True


        elif lang_rew_mode == 4: # 4. event det
            lang_network_dictpath = os.path.join(get_original_cwd(), cfg.rl_data_files.lang_rew_module_dictpath_event_det)
            langrewclass = LangRewEventDet
            is_goyal = False
            is_binary_cls = False

        elif lang_rew_mode == 5: # 5. event det hard neg
            lang_network_dictpath = os.path.join(get_original_cwd(), cfg.rl_data_files.lang_rew_module_dictpath_event_det_hard_negative)
            langrewclass = LangRewEventDet
            is_goyal = False
            is_binary_cls = False


        else:
            raise NotImplementedError

        if lang_rew_mode in [3]:
            cfg = OmegaConf.load(os.path.join(get_original_cwd(),'config/goyal_config.yaml'))
            lang_network = langrewclass(cfg).to(cfg.device_info.device)
            lang_network.load_state_dict(torch.load(lang_network_dictpath))
            lang_network = lang_network.half()
            lang_network.eval()
        else:
            # ------ Binary CLassification Ver Config ---------
            if cfg.lang_rew_shaping_params.norm_layer == "rmsnorm":
                norm_layer = RMSNorm
            elif cfg.lang_rew_shaping_params.norm_layer == "identity":
                norm_layer = torch.nn.Identity
            elif cfg.lang_rew_shaping_params.norm_layer == "batchnorm":
                norm_layer = torch.nn.BatchNorm1d
            elif cfg.lang_rew_shaping_params.norm_layer == "layernorm":
                norm_layer = torch.nn.LayerNorm
            else:
                raise NotImplementedError("no such norm layer")

            if cfg.data_files.pretrain_visual_encoder_use_object_detec:
                pretrain_visual_encoder_dict_path = os.path.join(get_original_cwd(),
                                                                 cfg.data_files.pretrain_visual_encoder_dict_path_with_object_detec)
            else:
                pretrain_visual_encoder_dict_path = os.path.join(get_original_cwd(),
                                                                 cfg.data_files.pretrain_visual_encoder_dict_path_without_object_detec)
            pretrain_action_predictor_dict_path = os.path.join(get_original_cwd(),
                                                               cfg.data_files.pretrain_action_predictor_dict_path)

            config_dict = dict(
                lw_img_size=cfg.CONSTANT.LW_RESIZE,
                sentence_dim=cfg.CONSTANT.SEN_EMB_DIM,
                visual_conv_depths=cfg.visual_encoder_params.conv_depths,
                visual_transformer_depth=cfg.visual_encoder_params.transformer_depths,
                visual_transformer_dim=cfg.visual_encoder_params.transformer_dims,
                visual_atten_head_num=cfg.visual_encoder_params.attn_head_num,
                visual_drop_rate=0,
                visual_encoder_dict_path=pretrain_visual_encoder_dict_path,
                action_predictor_dict_path=pretrain_action_predictor_dict_path,
                transformer_dim=cfg.lang_rew_shaping_params.transformer_dim,
                transformer_depth=cfg.lang_rew_shaping_params.transformer_depth,
                dropout_rate=cfg.lang_rew_shaping_params.dropout_rate,
                norm_layer=norm_layer,
                freeze_torso=cfg.lang_rew_shaping_params.freeze_torso,
                use_relative_offset=cfg.lang_rew_shaping_params.use_relative_offset,
                use_action_prediction=cfg.lang_rew_shaping_params.use_action_prediction,
                device=cfg.device_info.device
            )
            action_predictor_config_dict = dict(
                ACTION_SIZE=cfg.CONSTANT.N_ACTIONS,
                device=cfg.device_info.device,
                img_size=cfg.CONSTANT.LW_RESIZE,
                conv_depths=cfg.visual_action_predictor_params.conv_depths,
                conv_dims=cfg.visual_action_predictor_params.conv_dims,
            )

            # ------ End Binary CLassification Ver Config ---------
            lang_network = langrewclass(**config_dict, action_predictor_config_dict=action_predictor_config_dict).to(
                cfg.device_info.device)
            lang_network.load_state_dict(torch.load(lang_network_dictpath))
            lang_network = lang_network.half()
            lang_network.eval()

        # add attributes
        lang_network.is_goyal = is_goyal
        lang_network.is_binary_cls = is_binary_cls
        if is_goyal:
            lang_network.device = cfg.device_info.device
        lang_network.share_memory()

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

    if want_lang_rew:
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

    if want_lang_rew:
        return lang_network, walkthrough_raw_sentence, walkthrough_embds, task_saved_state, task_difficulty
    else:
        return None, [], None, task_saved_state, task_difficulty



