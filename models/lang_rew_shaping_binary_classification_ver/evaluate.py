import copy
import random
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from random import seed, shuffle
from hydra.utils import get_original_cwd

from backbones import RMSNorm
from dataloader import get_loader_sent_emb, get_hard_negative_dataloader
from models.lang_rew_shaping_binary_classification_ver.reward_shaping_module import RewardShapingModule
import torch
import os
from optimizer import get_optims_and_scheduler
from tqdm import tqdm
from pathlib import Path
import logging
from einops import reduce, repeat, rearrange
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

# this will produce confusion matrix
@torch.no_grad()
def evaluate(model, data_loader, device, tag):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device, non_blocking = True)
    accu_loss = torch.zeros(1).to(device, non_blocking=True)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    confusion_pred = []
    confusion_actual = []
    for data in data_loader:
        # labels : -> [B]
        *data, labels = model.handle_data(data)

        sample_num += labels.shape[0]

        # pred : -> [B, 2]
        pred = model(data)

        pred_classes = torch.max(pred, dim=1)[1]

        confusion_actual.extend(labels.detach().cpu().numpy())
        # handle confusion pred
        confusion_pred.extend(pred_classes.detach().cpu().numpy())

        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = "[evaluation] val loss: {:.3f}, val acc: {:.3f}".format(
            accu_loss.item() / sample_num,
            accu_num.item() / sample_num
        )
    # constant for classes
    classes = ('Positive', 'Negative')
    cf_matrix = confusion_matrix(confusion_actual, confusion_pred) # actual -> row; pred -> col
    temp = copy.deepcopy(cf_matrix)
    cf_matrix[0][0] = temp[1][1]
    cf_matrix[1][1] = temp[0][0]
    cf_matrix[0][1] = temp[1][0]
    cf_matrix[1][0] = temp[0][1]

    true_positive = cf_matrix[0][0]
    false_positive = cf_matrix[1][0]
    false_negative = cf_matrix[0][1]
    true_negative = cf_matrix[1][1]
    precision_score = true_positive / float(true_positive + false_positive)
    recall_score = true_positive / float(true_positive + false_negative)
    true_positive_rate = recall_score
    true_negative_rate = true_negative / float(true_negative + false_positive)
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    print(cf_matrix)
    df_data = normalize(cf_matrix, norm='l1')
    # df_data = cf_matrix
    df_cm = pd.DataFrame(df_data, index=[i for i in classes],
                         columns=[i for i in classes])
    fig, ax = plt.subplots(figsize=(7,5))
    sn.heatmap(df_cm, annot=True)
    ax.text(x=0.5, y=1.1, s='Confusion Matrix', fontsize=16, weight='bold', ha='center', va='bottom',
            transform=ax.transAxes)
    ax.text(x=0.5, y=1.05, s=f'precision: {precision_score:.2%}, recall: {recall_score:.2%}, balanced accuracy: {balanced_accuracy:.2%}', fontsize=8, alpha=0.75, ha='center',
            va='bottom', transform=ax.transAxes)

    plt.savefig(os.path.join(get_original_cwd(), f'confusion_matrix_{tag}.png'))

    return accu_loss.item() / sample_num, accu_num.item() / sample_num

@hydra.main(version_base=None, config_path="../../config", config_name="lang_rew_module")
def main(cfg: DictConfig):
    # print config
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)


    # model_path = os.path.join(get_original_cwd(),
    #                           'saved_model/trained_model/bi_cls-obj_det-T-rel_off-T-act_pred-T-neg_rat-0.7.pth')
    # tag = "bi_cls-obj_det-T-rel_off-T-act_pred-T-neg_rat-0.7"

    model_path = os.path.join(get_original_cwd(),
                              'saved_model/trained_model/bi_cls-obj_det-T-rel_off-T-act_pred-T-neg_rat-1.0.pth')
    tag = "bi_cls-obj_det-T-rel_off-T-act_pred-T-neg_rat-1.0"

    tag += f'-{cfg.data_files.task_level}-{cfg.data_files.no_hard_instruction}'

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

    # check model mode
    reward_shaping_module = RewardShapingModule(**config_dict,
                                                action_predictor_config_dict=action_predictor_config_dict).to(
        cfg.device_info.device)  # param (hidden_dim, sentence_dim, device, img_size, visual_encoder_cfg=None)



    reward_shaping_module.load_state_dict(torch.load(model_path))
    reward_shaping_module.eval()

    test_dataloader, _ = get_hard_negative_dataloader(cfg, is_train=False, is_validation=True)
    loss_valid, acc_valid = evaluate(model=reward_shaping_module,
                                     data_loader=test_dataloader,
                                     device=cfg.device_info.device,
                                     tag = tag,
                                     )
    log.info("VL: {:.2f} \t VA: {:.2%}".format(loss_valid, acc_valid))



if __name__ == '__main__':
    main()
