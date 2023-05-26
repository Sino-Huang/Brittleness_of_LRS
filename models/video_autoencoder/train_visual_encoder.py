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

from custom_utils import wandb_init, readimage, readVideo, save_to_video
from dataloader import get_pretrain_video_dataloader
from video_autoencoder import Video_Autoencoder
import torch
import os
from optimizer import get_optims_and_scheduler
from tqdm import tqdm
from pathlib import Path
import wandb

log = logging.getLogger(__name__)


# ------------------ training function -------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, batch_size, window_size, img_size, CLIP_LENGTH):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device, non_blocking=True)
    optimizer.zero_grad()
    mean = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]
    std = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    mean = torch.asarray(np.asarray(mean, dtype=np.float32)).to(device)
    std = torch.asarray(np.asarray(std, dtype=np.float32)).to(device)
    mean = repeat(mean, 'c -> b 1 c w h', b=batch_size, w=img_size, h=img_size)
    std = repeat(std, 'c -> b 1 c w h', b=batch_size, w=img_size, h=img_size)
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for data in data_loader:
        data, labels = model.handle_data(data, WINDOW_SIZE=window_size, CLIP_LENGTH=CLIP_LENGTH)
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # selected_frame_ind =
        pred_img_normalised = model(data) # ndarray type (batch, frame 3, w, h) 0-1 float
        pred_img = ((pred_img_normalised * 255.0) - mean) / std
        loss = loss_function(pred_img, labels)
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

@hydra.main(version_base=None, config_path="../../config", config_name="video_autoencoder_pretrain")
def main(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)
    log.info("--------Model Config---------")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("-----End of Model Config-----")
    jobname = HydraConfig.get().job.name
    if cfg.data_files.use_wandb:
        wandb_init(cfg_raw, jobname, ["visual_encoder"])

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

    # create save model folder
    # create the folder at root
    Path(os.path.join(get_original_cwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(get_original_cwd(), cfg.visual_encoder_params.saved_path_folder)).mkdir(parents=True,
                                                                                              exist_ok=True)
    # create folder at runtime
    Path(os.path.join(os.getcwd(), cfg.data_files.saved_path_parent_folder)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(os.getcwd(), cfg.visual_encoder_params.saved_path_folder)).mkdir(parents=True, exist_ok=True)


    # init visual model
    rl_img_size = cfg.CONSTANT.RL_RESIZE
    img_size = cfg.CONSTANT.LW_RESIZE
    conv_depths = cfg.visual_encoder_params.conv_depths
    transformer_depths = cfg.visual_encoder_params.transformer_depths
    transformer_dims = cfg.visual_encoder_params.transformer_dims
    attn_head_num = cfg.visual_encoder_params.attn_head_num
    drop_rate = cfg.visual_encoder_params.drop_rate

    visual_model = Video_Autoencoder(
        rl_img_size = rl_img_size,
        img_size = img_size,
        conv_depths = conv_depths,
        transformer_depths = transformer_depths,
        transformer_dims = transformer_dims,
        attn_head_num = attn_head_num,
        drop_rate = drop_rate,
    ).to(cfg.device_info.device)
    visual_model.train()

    if cfg.data_files.use_wandb:
        # wandb watch model
        wandb.watch(visual_model)

    # load training data
    data_path = os.path.join(get_original_cwd(), cfg.data_files.raw_video_screenshot_file_input)
    data_path_2 = os.path.join(get_original_cwd(), cfg.data_files.raw_video_screenshot_file_target)
    train_dataloader, steps_per_epoch = get_pretrain_video_dataloader(data_path,
                                                                      data_path_2,
                                                                      batch_size=cfg.visual_encoder_params.batch_size,
                                                                      sequence_length=cfg.CONSTANT.CLIP_LENGTH,
                                                                      window_size=cfg.CONSTANT.PRETRAIN_WINDOW_SIZE,
                                                                      first_resize=cfg.CONSTANT.LW_RESIZE,
                                                                      second_resize=cfg.CONSTANT.RL_RESIZE)

    best_val_loss = 1000000
    best_epoch = 0

    # Optimizer and lr scheduler
    optimizer, lr_scheduler = get_optims_and_scheduler(visual_model, cfg, steps_per_epoch)
    # Training code

    for epoch in range(cfg.visual_encoder_params.visual_n_epoch):
        # train
        loss_train = train_one_epoch(model=visual_model,
                                     optimizer=optimizer,
                                     data_loader=train_dataloader,
                                     device=cfg.device_info.device,
                                     epoch=epoch,
                                     lr_scheduler=lr_scheduler,
                                     batch_size=cfg.visual_encoder_params.batch_size,
                                     window_size=cfg.CONSTANT.PRETRAIN_WINDOW_SIZE,
                                     img_size=cfg.CONSTANT.RL_RESIZE,
                                     CLIP_LENGTH = cfg.CONSTANT.CLIP_LENGTH)



        if loss_train < best_val_loss:
            # mkdir if not exist
            visual_model.save(os.path.join(os.getcwd(), cfg.visual_encoder_params.visual_encoder_state_dict_path))
            # add soft link to main folder
            softlink_path = Path(
                os.path.join(get_original_cwd(), cfg.visual_encoder_params.visual_encoder_state_dict_path))
            # softlink path needs to be complete
            softlink_path.unlink(missing_ok=True)  # unlink (i.e., delete) to avoid FileExistError
            softlink_path.symlink_to(
                os.path.join(os.getcwd(), cfg.visual_encoder_params.visual_encoder_state_dict_path))
            best_val_loss = loss_train
            best_epoch = epoch

        log.info('Epoch: {:d} \t TL: {:5f} \t BE: {:}\t BL: {:}'
                 .format(epoch, loss_train,
                         best_epoch, best_val_loss))

        if cfg.data_files.use_wandb:
            # wandb log
            # time for wandb to log
            wandb.log({
                "train_loss": loss_train,
                "best_train_loss": best_val_loss,
            })

        # Construct Eval Video
        mean = torch.asarray(np.asarray(eval(cfg.CONSTANT.IMG_MEAN), dtype=np.float32)).to(cfg.device_info.device)
        std = torch.asarray(np.asarray(eval(cfg.CONSTANT.IMG_STD), dtype=np.float32)).to(cfg.device_info.device)
        mean = repeat(mean, 'c -> 1 c w h', w=cfg.CONSTANT.RL_RESIZE, h=cfg.CONSTANT.RL_RESIZE)
        std = repeat(std, 'c -> 1 c w h', w=cfg.CONSTANT.RL_RESIZE, h=cfg.CONSTANT.RL_RESIZE)
        visual_model.eval()
        videopath = os.path.join(get_original_cwd(), cfg.visual_encoder_params.video_example_path)
        window_size = cfg.CONSTANT.PRETRAIN_WINDOW_SIZE
        vid_data = readVideo(videopath, mean=eval(cfg.CONSTANT.IMG_MEAN), std=eval(cfg.CONSTANT.IMG_STD), windowsize=window_size, resize=cfg.CONSTANT.LW_RESIZE)
        label = readVideo(videopath, mean=eval(cfg.CONSTANT.IMG_MEAN), std=eval(cfg.CONSTANT.IMG_STD), windowsize=window_size, resize=cfg.CONSTANT.RL_RESIZE)
        # add batch dim
        vid_data = rearrange(vid_data, "f c w h -> 1 f c w h")
        label = rearrange(label, "f c w h -> 1 f c w h")
        # ready for model
        vid_data = torch.asarray(vid_data).float().to(cfg.device_info.device)
        label = torch.asarray(label).float().to(cfg.device_info.device)

        pred_img_normalised = visual_model(vid_data)
        # squeeze
        pred_img_normalised = rearrange(pred_img_normalised, "1 f c w h -> f c w h")
        pred_img = ((pred_img_normalised * 255.0) - mean) / std

        loss = torch.nn.MSELoss()(pred_img, label.squeeze(0))
        print(f"loss for picked sample is {loss.item()}")

        pred_img_output = pred_img_normalised.cpu().detach().numpy() * 255.0
        pred_img_output = rearrange(pred_img_output, 'f c w h -> f w h c')
        pred_img_output = np.round(pred_img_output).astype(np.uint8)
        pred_img_output = np.clip(pred_img_output, 0, 255)

        save_to_video(pred_img_output, os.path.join(get_original_cwd(), f"recon_{epoch}.mp4"), fps=6.25)
        visual_model.train()



if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
