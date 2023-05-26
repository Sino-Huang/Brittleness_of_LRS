import math
import random
import sys

import cv2
import hydra
import logging

from einops import rearrange, repeat
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import numpy as np
from hydra.utils import get_original_cwd, HydraConfig

from custom_utils import wandb_init, readimage
from dataloader import get_pretrain_video_action_dataloader
from video_action_predictor import Video_Action_Predictor
import torch
import os
from optimizer import get_optims_and_scheduler
from tqdm import tqdm
from pathlib import Path
import wandb

log = logging.getLogger(__name__)


# ------------------ training function -------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, batch_size, img_size):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device, non_blocking=True)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for data in data_loader:
        data, labels = model.handle_data(data)
        pred = model(data)
        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
        step += 1
    return accu_loss.item() / (step + 1)


# -------------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.MSELoss()

    model.eval()

    accu_loss = torch.zeros(1).to(device, non_blocking=True)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for data in data_loader:
        data, labels = model.handle_data(data)

        sample_num += labels.shape[0]

        pred = model(data)

        loss = loss_function(pred, labels)

        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(
            epoch,
            accu_loss.item() / sample_num,
        )
    return accu_loss.item() / sample_num

# -------------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../config", config_name="video_action_pretrain")
def main(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")
    jobname = HydraConfig.get().job.name
    if cfg.data_files.use_wandb:
        wandb_init(cfg_raw, jobname, ["video_action_predictor"])

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    # create save model folder
    # create the folder at root
    Path(os.path.join(get_original_cwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(get_original_cwd(), cfg.visual_action_predictor_params.saved_path_folder)).mkdir(parents=True,
                                                                                              exist_ok=True)
    # create folder at runtime
    Path(os.path.join(os.getcwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(os.getcwd(), cfg.visual_action_predictor_params.saved_path_folder)).mkdir(parents=True, exist_ok=True)

    action_predictor_config_dict = dict(
        ACTION_SIZE=cfg.CONSTANT.N_ACTIONS,
        device=cfg.device_info.device,
        img_size=cfg.CONSTANT.LW_RESIZE,
        conv_depths=cfg.visual_action_predictor_params.conv_depths,
        conv_dims=cfg.visual_action_predictor_params.conv_dims,
    )
    # init visual model
    visual_model = Video_Action_Predictor(**action_predictor_config_dict).to(cfg.device_info.device)
    visual_model.train()

    if cfg.data_files.use_wandb:
        # wandb watch model
        wandb.watch(visual_model)

    # load training data
    train_dataloader, steps_per_epoch = get_pretrain_video_action_dataloader(cfg, True)
    test_dataloader, _ = get_pretrain_video_action_dataloader(cfg, False)

    best_val_loss = 1000000
    best_epoch = 0

    # Optimizer and lr scheduler
    optimizer, lr_scheduler = get_optims_and_scheduler(visual_model, cfg, steps_per_epoch)
    # Training code

    for epoch in range(cfg.visual_action_predictor_params.visual_n_epoch):
        # train
        loss_train = train_one_epoch(model=visual_model,
                                     optimizer=optimizer,
                                     data_loader=train_dataloader,
                                     device=cfg.device_info.device,
                                     epoch=epoch,
                                     lr_scheduler=lr_scheduler,
                                     batch_size=cfg.visual_action_predictor_params.batch_size,
                                     img_size=cfg.CONSTANT.LW_RESIZE)

        # validate
        loss_valid = evaluate(model=visual_model,
                                         data_loader=test_dataloader,
                                         device=cfg.device_info.device,
                                         epoch=epoch
                                         )



        if loss_valid < best_val_loss:
            # mkdir if not exist
            visual_model.save(os.path.join(os.getcwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path))
            # add soft link to main folder
            softlink_path = Path(
                os.path.join(get_original_cwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path))
            # softlink path needs to be complete
            softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
            softlink_path.symlink_to(
                os.path.join(os.getcwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path))
            best_val_loss = loss_valid
            best_epoch = epoch


        if cfg.data_files.use_wandb:
            # wandb log
            # time for wandb to log
            wandb.log({
                "train_loss": loss_train,
                "valid_loss": loss_valid,
                "best_valid_loss": best_val_loss
            })

        log.info('Epoch: {:d} \t TL: {:5f} \t VL: {:5f} \t BE: {:}\t BVL: {:}'
                 .format(epoch, loss_train, loss_valid,
                         best_epoch, best_val_loss))

    # end, save the final state dict
    # mkdir if not exist
    visual_model.save(os.path.join(os.getcwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path_final))
    # add soft link to main folder
    softlink_path = Path(os.path.join(get_original_cwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path_final))
    # softlink path needs to be complete
    softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
    softlink_path.symlink_to(os.path.join(os.getcwd(), cfg.visual_action_predictor_params.visual_encoder_state_dict_path_final))
    # finish wandb automatically for multirun
    if cfg.data_files.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
