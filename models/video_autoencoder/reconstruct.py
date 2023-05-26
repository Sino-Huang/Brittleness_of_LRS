import math
import random
import sys

import cv2
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import numpy as np
from random import seed, shuffle
from hydra.utils import get_original_cwd, HydraConfig
from video_autoencoder import Video_Autoencoder
import torch
import os
from optimizer import get_optims_and_scheduler
from tqdm import tqdm
from pathlib import Path
from custom_utils import readVideo, save_to_video
from einops import rearrange, repeat, reduce
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

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


    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

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
    visual_model.eval()

    visual_model.load_state_dict(
        torch.load(os.path.join(get_original_cwd(), cfg.visual_encoder_params.visual_encoder_state_dict_path)))
    visual_model.eval()



    # load val data
    # valid_dataloader = get_pretrain_vit_test_dataloader(cfg)
    loss_function = torch.nn.MSELoss()

    videopath = os.path.join(get_original_cwd(), cfg.visual_encoder_params.video_example_path)
    IMG_MEAN = eval(cfg.CONSTANT.IMG_MEAN)
    IMG_STD = eval(cfg.CONSTANT.IMG_STD)
    mean = torch.asarray(np.asarray(IMG_MEAN, dtype=np.float32)).to(cfg.device_info.device)
    std = torch.asarray(np.asarray(IMG_STD, dtype=np.float32)).to(cfg.device_info.device)
    mean = repeat(mean, 'c -> 1 c w h', w=cfg.CONSTANT.RL_RESIZE, h=cfg.CONSTANT.RL_RESIZE)
    std = repeat(std, 'c -> 1 c w h', w=cfg.CONSTANT.RL_RESIZE, h=cfg.CONSTANT.RL_RESIZE)


    window_size = cfg.CONSTANT.PRETRAIN_WINDOW_SIZE
    vid_data = readVideo(videopath, mean=IMG_MEAN, std=IMG_STD, windowsize=window_size, resize=cfg.CONSTANT.LW_RESIZE)
    label = readVideo(videopath, mean=IMG_MEAN, std=IMG_STD, windowsize=window_size, resize=cfg.CONSTANT.RL_RESIZE)
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

    loss = loss_function(pred_img, label.squeeze(0))
    print(f"loss for picked sample is {loss.item()}")

    pred_img_output = pred_img_normalised.cpu().detach().numpy() * 255.0
    pred_img_output = rearrange(pred_img_output, 'f c w h -> f w h c')
    pred_img_output = np.round(pred_img_output).astype(np.uint8)
    pred_img_output = np.clip(pred_img_output, 0, 255)

    save_to_video(pred_img_output, os.path.join(get_original_cwd(), "recon_eval.mp4"), fps=6.25)
    torch.save(visual_model.torsonet.state_dict(), os.path.join(get_original_cwd(), "torsonet_pretrained.pth"))

if __name__ == '__main__':
    main()