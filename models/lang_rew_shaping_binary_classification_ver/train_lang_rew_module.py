import copy
import math
import random
import sys
import time

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
from random import seed, shuffle
from hydra.utils import get_original_cwd, HydraConfig
from sklearn.metrics import confusion_matrix

from custom_utils import wandb_init, generalized_time_iou
from dataloader import get_hard_negative_dataloader
from reward_shaping_module import RewardShapingModule
import torch
import os
from optimizer import get_optims_and_scheduler
from backbones import RMSNorm
from tqdm import tqdm
from pathlib import Path
import logging
from einops import reduce, repeat, rearrange

log = logging.getLogger(__name__)

# ------------------ training function -------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    # classification_loss_function = torch.nn.BCEWithLogitsLoss()                                        # CHANGE
    classification_loss_function = torch.nn.CrossEntropyLoss()

    accumu_loss = torch.zeros(1).to(device, non_blocking=True)
    accu_num = torch.zeros(1).to(device, non_blocking=True)

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for data in data_loader:
        *data, labels = model.handle_data(data)
        sample_num += labels.shape[0]

        pred = model(data) # shape [B,3]
        # label shape: torch.Size([32])
        # pred_classes = torch.round(pred).long() # this output the ind -> 1 is matched, 0 is non-matched # CHANGE
        pred_classes = torch.max(pred, dim=1)[1]  # this output the max indices

        # accu_num += torch.eq(pred_classes, labels.unsqueeze(-1)).sum()                                  # CHANGE
        accu_num += torch.eq(pred_classes, labels).sum()

        # loss = classification_loss_function(pred, labels.unsqueeze(-1).float())                         # CHANGE
        loss = classification_loss_function(pred, labels)

        loss.backward()
        accumu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.5f}, acc: {:.5f}, lr: {:.5f}".format(
            epoch,
            accumu_loss.item() / sample_num,
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
    return accumu_loss.item() / sample_num, accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device, non_blocking=True)
    accu_loss = torch.zeros(1).to(device, non_blocking=True)

    sample_num = 0
    confidence_threshold = 0.2
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
    cf_matrix = confusion_matrix(confusion_actual, confusion_pred)  # actual -> row; pred -> col
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


    return accu_loss.item() / sample_num, balanced_accuracy

# -------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../config", config_name="lang_rew_module")
def main(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")
    jobname = HydraConfig.get().job.name
    if cfg.data_files.use_wandb:
        wandb_init(cfg_raw, jobname, ["reward_shaping_module_Nov", "lang_rew_binary_cls_ver"])

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    # create save model folder
    # create the folder at root
    Path(os.path.join(get_original_cwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(get_original_cwd(), cfg.data_files.saved_path_folder)).mkdir(parents=True, exist_ok=True)
    # create folder at runtime
    Path(os.path.join(os.getcwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(os.getcwd(), cfg.data_files.saved_path_folder)).mkdir(parents=True, exist_ok=True)

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
        pretrain_visual_encoder_dict_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_visual_encoder_dict_path_without_object_detec)
    pretrain_action_predictor_dict_path= os.path.join(get_original_cwd(), cfg.data_files.pretrain_action_predictor_dict_path)

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
    reward_shaping_module = RewardShapingModule(**config_dict, action_predictor_config_dict = action_predictor_config_dict).to(cfg.device_info.device)  # param (hidden_dim, sentence_dim, device, img_size, visual_encoder_cfg=None)

    if cfg.lang_rew_shaping_params.mode == "new":
        reward_shaping_module.train()
    elif cfg.lang_rew_shaping_params.mode == "continue":
        # load the state dict
        reward_shaping_module.load_state_dict(torch.load(os.path.join(get_original_cwd(), cfg.data_files.pretrain_normal_negative_lang_rew_module_dict_path)))
        reward_shaping_module.train()
    else:
        raise NotImplementedError

    if cfg.data_files.use_wandb:
        # wandb watch model
        wandb.watch(reward_shaping_module)

    # init the best val acc and best epoch
    best_val_loss = 10000.0
    best_train_loss = 10000.0
    best_val_acc = 0.0
    best_train_acc = 0.0

    best_val_epoch = 0
    best_train_epoch = 0

    # Data loader

    train_dataloader, steps_per_epoch = get_hard_negative_dataloader(cfg, is_train=True, is_validation=False)
    validation_dataloader, _ = get_hard_negative_dataloader(cfg, is_train=True, is_validation=True)

    # Optimizer and lr scheduler
    optimizer, lr_scheduler = get_optims_and_scheduler(reward_shaping_module, cfg, steps_per_epoch)



    # Training code

    for epoch in range(cfg.lang_rew_shaping_params.n_epochs):
        # train
        loss_train, acc_train = train_one_epoch(model=reward_shaping_module,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=cfg.device_info.device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                )

        # validate
        loss_valid, acc_valid = evaluate(model=reward_shaping_module,
                                         data_loader=validation_dataloader,
                                         device=cfg.device_info.device,
                                         epoch=epoch,
                                         )

        if cfg.data_files.save_every_epoch:
            reward_shaping_module.save(os.path.join(cfg.data_files.saved_path_folder, f"reward_shaping_module_epoch_{epoch}.pth"))



        if loss_valid <= best_val_loss: # we store the best valid loss rather than best valid acc
            # mkdir if not exist
            reward_shaping_module.save(os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path))
            # add soft link to main folder
            # softlink_path = Path(os.path.join(get_original_cwd(), cfg.data_files.reward_shaping_module_state_dict_path))
            # # softlink path needs to be complete
            # softlink_path.unlink(missing_ok=True) # unlink (i.e., delete) to avoid FileExistError
            # time.sleep(0.1) # unlink require sometime
            # softlink_path.symlink_to(os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path))
            best_val_loss = loss_valid

        if loss_train <= best_train_loss: # we store the best train loss
            # mkdir if not exist
            reward_shaping_module.save(
                os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_train))
            # add soft link to main folder
            # softlink_path = Path(
            #     os.path.join(get_original_cwd(), cfg.data_files.reward_shaping_module_state_dict_path_train))
            # # softlink path needs to be complete
            # softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
            # time.sleep(0.1) # unlink require sometime
            # softlink_path.symlink_to(
            #     os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_train))
            best_train_loss = loss_train

        if acc_train >= best_train_acc:
            # mkdir if not exist
            reward_shaping_module.save(
                os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_train_acc))
            # add soft link to main folder
            # softlink_path = Path(
            #     os.path.join(get_original_cwd(), cfg.data_files.reward_shaping_module_state_dict_path_train_acc))
            # # softlink path needs to be complete
            # softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
            # time.sleep(0.1) # unlink require sometime
            # softlink_path.symlink_to(
            #     os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_train_acc))
            best_train_acc = acc_train
            best_train_epoch = epoch



        if acc_valid >= best_val_acc:
            # mkdir if not exist
            reward_shaping_module.save(os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_acc))
            # add soft link to main folder
            # softlink_path = Path(os.path.join(get_original_cwd(), cfg.data_files.reward_shaping_module_state_dict_path_acc))
            # # softlink path needs to be complete
            # softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
            # softlink_path.symlink_to(os.path.join(os.getcwd(), cfg.data_files.reward_shaping_module_state_dict_path_acc))
            best_val_acc = acc_valid
            best_val_epoch = epoch



        # Log the result.
        if cfg.data_files.use_wandb:
            # wandb log
            wandb.log({
                "train_loss" : loss_train,
                "valid_loss" : loss_valid,
                "train_acc" : acc_train,
                "valid_acc" : acc_valid,
                "learning_rate" : optimizer.param_groups[0]["lr"],
                "best_valid_loss": best_val_loss,
                "best_train_loss": best_train_loss,
                'best train acc': best_train_acc,
                'best valid acc': best_val_acc
            })
        log.info('Epoch: {:d} \t TL: {:5f} \t VL: {:5f} \t TAcc: {:.5f} \t VAcc: {:.5f}\t BVE: {:d}\t BVL: {:.5f}\t BTE: {:d}\t BTL: {:.5f}'
                 .format(epoch, loss_train, loss_valid, acc_train, acc_valid,
                         best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))

    # finish wandb automatically for multirun
    if cfg.data_files.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()