import random

import hydra
import numpy as np
import sys
import pickle
from random import shuffle, seed, randint
import math
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra.utils import get_original_cwd
from einops import reduce, rearrange, repeat
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence

from backbones import Gated_XAtten_Dense_Block, RMSNorm, RotaryEmbedding, apply_rotary_pos_emb, TorsoNetwork, \
    ConvNeXtMini
from custom_utils.utils import get_relative_offset
from models.video_action_predictor import Video_Action_Predictor


# image shape: torch.Size([32, dif_length, 3, 84, 84])
# image are normalised by ImageNet mean and std
# lang shape: torch.Size([32, 768]) for sentence emb
# label shape: torch.Size([32])

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
            nn.Linear(self.middle_dim, self.output_dim),
            norm_layer(self.output_dim),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class EncLSTM(nn.Module):
    '''
    Batch first LSTM
    '''
    def __init__(self, in_feature_dim, hidden_dim, dropout_rate, n_layers = 2):
        super().__init__()
        self.n_layer = n_layers
        self.dropout_rate = dropout_rate
        self.lstm_size = hidden_dim
        self.in_feature_size = in_feature_dim
        self.lstm = nn.GRU(input_size=self.in_feature_size,
                           hidden_size=self.lstm_size,
                           num_layers=self.n_layer,
                           batch_first=True,
                           dropout= self.dropout_rate,
                           bidirectional=False)

        self.norm = nn.LayerNorm(normalized_shape=self.lstm_size, eps=1e-6)

    def forward(self, x): # shape (32(batch), 128(length), 50(feature))
        x, hn = self.lstm(x) # shape (32(batch), 128(length), 128(hidden_size))
        x = x[:,-1,:] # shape (32(batch), 128(hidden_size))
        x = self.norm(x)
        return x


class RewardShapingModule(nn.Module):
    def __init__(self,
                 lw_img_size, sentence_dim,
                 visual_conv_depths, visual_transformer_depth,
                 visual_transformer_dim, visual_atten_head_num, visual_drop_rate,
                 visual_encoder_dict_path,
                 action_predictor_dict_path,
                 transformer_dim, transformer_depth, dropout_rate, norm_layer, freeze_torso,
                 use_relative_offset,
                 use_action_prediction,
                 device,
                 action_predictor_config_dict):
        super().__init__()

        self.device = device
        self.use_relative_offset = use_relative_offset
        self.use_action_prediction = use_action_prediction
        # output (batch, frame, lstm dims)
        #  ----------  Pretraining visual encoder --------------
        self.obs_torso_encoder = TorsoNetwork(
            img_size=lw_img_size,
            conv_depths=visual_conv_depths,
            transformer_depths=visual_transformer_depth,
            transformer_dims=visual_transformer_dim,
            attn_head_num=visual_atten_head_num,
            drop_rate=visual_drop_rate,
            conv_dims=[96,192,256],
        )
        self.obs_torso_encoder.load_state_dict(
            torch.load(visual_encoder_dict_path),
            strict=True
        )

        if self.use_action_prediction:
            self.action_predictor = Video_Action_Predictor(**action_predictor_config_dict)
            self.action_predictor.load_state_dict(
                torch.load(action_predictor_dict_path),
                strict=True
            )


        # need to update weights of the pretrained model because we have attention between sentence embedding and the video
        if freeze_torso:
            for param in self.obs_torso_encoder.parameters():
                param.requires_grad = False
            self.obs_torso_encoder.eval()

            if self.use_action_prediction:

                for param in self.action_predictor.parameters():
                    param.requires_grad = False
                self.action_predictor.eval()

        else:
            for param in self.obs_torso_encoder.parameters():
                param.requires_grad = True
            self.obs_torso_encoder.train()

            if self.use_action_prediction:

                for param in self.action_predictor.parameters():
                    param.requires_grad = True
                self.action_predictor.train()

        t = torch.rand((1, 1, 3, lw_img_size, lw_img_size)) # visual_encoder input shape (batch, frame, channel, w, h)
        t_out = self.obs_torso_encoder(t) # visual_encoder output shape (batch, frame, lstm_dim)
                                             # action predictor input shape (batch, LEN(2), channel, W, H)
                                             # action predictor output shape (batch, action_size)
        visual_output_dim = t_out.shape[-1]
        if self.use_action_prediction:
            visual_output_dim += action_predictor_config_dict['ACTION_SIZE']
        print('visual output dim', visual_output_dim)

        # ---------------- End of Loading Pretrained visual encoder ---------------------


        # # --------- Transformer part (commend it out if you want to try mlp architecture) --------
        #
        # self.text_reduct = nn.Sequential( # input shape [batch, 768]
        #     MLPBlock(sentence_dim, transformer_dim, norm_layer=norm_layer),
        #     nn.GELU()
        # )
        #
        #
        # droppathrate = [x.item() for x in torch.linspace(0, dropout_rate,
        #                                                  transformer_depth)]  # stochastic depth decay
        # self.transformer = nn.ModuleList(
        #     [Gated_XAtten_Dense_Block(q_dim=transformer_dim,
        #                               kv_dim=visual_output_dim, depth=transformer_depth,
        #                               num_heads=8, drop_path_ratio=droppathrate[i]) for i in
        #      range(transformer_depth)]
        # )
        # self.after_transformer_norm = nn.LayerNorm(transformer_dim)
        #
        # # positional embedding
        # self.rotatory_embed_generator = RotaryEmbedding(
        #     transformer_dim // 8)  # position embedding dim will be reduced by attn head
        # self.pos_emb = self.rotatory_embed_generator(96)
        #
        # self.head = nn.Sequential(
        #     nn.Linear(transformer_dim, transformer_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(transformer_dim * 2, 3),
        #     nn.Dropout(dropout_rate),
        #     norm_layer(3),
        #
        # )
        #
        # # ------ End of Transformer Architecture ------

        # -------- MLP architecture ---------
        self.text_enc_mlp = nn.Sequential(
            MLPBlock(sentence_dim, transformer_dim, norm_layer=norm_layer),
            nn.GELU()
        )
        self.vid_enc_lstm = EncLSTM(visual_output_dim, transformer_dim, dropout_rate, transformer_depth)

        self.classifier = nn.Sequential(
            nn.Linear(2 * transformer_dim, 3 * transformer_dim),
            nn.GELU(),
            norm_layer(3 * transformer_dim),
            nn.Linear(3 * transformer_dim, 2)
        )
        # ----- End of MLP Architecture --------


    # # ---------Transformer version forward--------
    # def forward(self, x, mask=None):
    #     # mask shape (B, L) # mask 1 means we keep it during attention, 0 means we do not use it
    #     text, video = x # shape text:(batch, txt_dim), video:([batch, varying_frame_len, channel, w, h])
    #     B= video.shape[0]
    #     text = self.text_reduct(text) # shape [batch, hidden_dim]
    #     text = rearrange(text, 'b d -> b 1 d') # [batch, 1,  hidden_dim]
    #
    #     video_latent = self.obs_torso_encoder(video, mask=mask) # shape: [batch, varying_frame_len, output_dim]
    #
    #     if self.use_relative_offset:
    #         video_latent = get_relative_offset(video_latent)
    #     if self.use_action_prediction:
    #         #  video:([batch, varying_frame_len, channel, w, h])
    #         video_grayscale = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]
    #         # calculate diff
    #         video_diff = video_grayscale[:, 1:] - video_grayscale[:, :-1]  # shape (b, f-1, w, h)
    #         video_diff = rearrange(video_diff, 'b f w h -> (b f) 1 w h')
    #         action_latent = self.action_predictor(video_diff) # shape (big B, action)
    #         action_latent = rearrange(action_latent, '(b f) d -> b f d', b=B) # shape [batch, varying_frame_len, output_dim]
    #         if not self.use_relative_offset: # if not use relative offset, we need to add one more zeros in front
    #             z = torch.zeros((B, 1, action_latent.shape[-1]), dtype=action_latent.dtype).to(action_latent.device)
    #             action_latent = torch.cat([z, action_latent], dim = -2)
    #         video_latent = torch.cat([video_latent, action_latent], dim=-1)
    #     pos_emb = self.pos_emb[:video_latent.shape[-2]]
    #
    #     for block in self.transformer:
    #         text = block(text, video_latent, video_latent, mask, pos_emb, cross_pos_only=True, k_pos_only=True)
    #
    #
    #     text = self.after_transformer_norm(text)
    #     text = rearrange(text, 'b 1 d -> b d') # [batch, hidden_dim]
    #
    #     logit = self.head(text) # the text token
    #     logit = torch.hstack([torch.tanh(logit[:,[0]]), torch.sigmoid(logit[:,1:])]) # normalise IoU, event start and end
    #     return logit # shape [batch, 3] 3 are IoU, event_start, event_duration
    # # ---- End of Transformer version forward ----------

        # ----- MLP version forward ------
    def forward(self, x):
        text, video = x  # shape text:(batch, txt_dim), video:([batch, varying_frame_len, channel, w, h])
        B = text.shape[0]
        text_enc = self.text_enc_mlp(text)

        video_latent = self.obs_torso_encoder(video)  # shape: [batch, varying_frame_len, output_dim]
        if self.use_relative_offset:
            video_latent = get_relative_offset(video_latent)

        if self.use_action_prediction:
            #  video:([batch, varying_frame_len, channel, w, h])
            video_grayscale = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]
            # calculate diff
            video_diff = video_grayscale[:, 1:] - video_grayscale[:, :-1]  # shape (b, f-1, w, h)
            video_diff = rearrange(video_diff, 'b f w h -> (b f) 1 w h')
            action_latent = self.action_predictor(video_diff)  # shape (big B, action)
            action_latent = rearrange(action_latent, '(b f) d -> b f d',
                                      b=B)  # shape [batch, varying_frame_len, output_dim]
            if not self.use_relative_offset:  # if not use relative offset, we need to add one more zeros in front
                z = torch.zeros((B, 1, action_latent.shape[-1]), dtype=action_latent.dtype).to(action_latent.device)
                action_latent = torch.cat([z, action_latent], dim=-2)

            video_latent = torch.cat([video_latent, action_latent], dim=-1)

        video_latent = self.vid_enc_lstm(video_latent)

        text_action_video_concat = torch.cat((text_enc, video_latent),
                                             dim=-1)  # shape: ((32(batch), (out_feature)))

        logit = self.classifier(text_action_video_concat)
        # logit = torch.hstack(
        #     [torch.tanh(logit[:, [0]]), torch.sigmoid(logit[:, 1:])])  # normalise IoU, event start and end

        logit = torch.column_stack([torch.sigmoid(logit[:,0]), torch.tanh(logit[:,1])])
        return logit

    # ----- End of MLP version forward ------


    def save(self, path):
        r'''

        :param path: path to save model
        :return:
        '''
        torch.save(self.state_dict(), path)


    def handle_data(self, x):
        text = x[0]["lang"] # shape (batch, con_dim)
        video = x[0]["video"] # shape (batch, len, channel, w, h) normalised
        label = x[0]['label'] # shape (batch, 2)
        text = text.to(self.device, non_blocking = True)
        video = video.to(self.device, non_blocking = True)
        label = label.to(self.device, non_blocking = True)
        return [text, video, label]

@hydra.main(version_base=None, config_path="../../config", config_name="lang_rew_module")
def testnetwork(cfg: DictConfig):
    # print config
    cfg_raw = OmegaConf.to_container(cfg, resolve=True)  # OmegaConf is used for wandb
    cfg = DictConfig(cfg_raw)

    # set up random seed
    np.random.seed(cfg.CONSTANT.RANDOM_SEED)
    torch.manual_seed(cfg.CONSTANT.RANDOM_SEED)
    random.seed(cfg.CONSTANT.RANDOM_SEED)

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

    pretrain_visual_encoder_dict_path = os.path.join(get_original_cwd(), cfg.data_files.pretrain_visual_encoder_dict_path)

    config_dict = dict(
        lw_img_size=cfg.CONSTANT.LW_RESIZE,
        sentence_dim=cfg.CONSTANT.SEN_EMB_DIM,
        visual_conv_depths=cfg.visual_encoder_params.conv_depths,
        visual_transformer_depth=cfg.visual_encoder_params.transformer_depths,
        visual_transformer_dim=cfg.visual_encoder_params.transformer_dims,
        visual_atten_head_num=cfg.visual_encoder_params.attn_head_num,
        visual_drop_rate=0,
        visual_encoder_dict_path=pretrain_visual_encoder_dict_path,
        transformer_dim=cfg.lang_rew_shaping_params.transformer_dim,
        transformer_depth=cfg.lang_rew_shaping_params.transformer_depth,
        dropout_rate=cfg.lang_rew_shaping_params.dropout_rate,
        norm_layer=norm_layer,
        freeze_torso=cfg.lang_rew_shaping_params.freeze_torso,
        use_relative_offset=cfg.lang_rew_shaping_params.use_relative_offset,
        use_action_prediction=cfg.lang_rew_shaping_params.use_action_prediction,
        device=cfg.device_info.device
    )

    # check model mode
    reward_shaping_module = RewardShapingModule(**config_dict)  # param (hidden_dim, sentence_dim, device, img_size, visual_encoder_cfg=None)

    lang = torch.rand((32, 768), dtype=torch.float32)
    video = torch.rand((32, 14, 3, 84, 84), dtype=torch.float32)

    output = reward_shaping_module([lang, video])
    print(output.shape) # torch.Size([32, 3])

if __name__ == '__main__':
    testnetwork()