import random

import torch
import torch.nn as nn
import torch.functional as F
from einops.layers.torch import Rearrange
from nfnets import replace_conv, AGC, WSConv2d, ScaledStdConv2d
from backbones import ConvNeXtMini, Gated_XAtten_Dense_Block, RotaryEmbedding, ConvSimpleDecoder
from einops import reduce, repeat, rearrange

# # input has length 20, batch size 32 and dimension 128
# x = torch.FloatTensor(20, 32, 128).cuda()
#
# input_size, hidden_size = 128, 128
#
# rnn = SRU(input_size, hidden_size,
#     num_layers = 2,          # number of stacking RNN layers
#     dropout = 0.0,           # dropout applied between RNN layers
#     bidirectional = False,   # bidirectional RNN
#     layer_norm = False,      # apply layer normalization on the output of each layer
#     highway_bias = 0,        # initial bias of highway gate (<= 0)
#     rescale = True,          # whether to use scaling correction
# )
# rnn.cuda()
#
# output_states, c_states = rnn(x)      # forward pass
#
# # output_states is (length, batch size, number of directions * hidden size)
# # c_states is (layers, batch size, number of directions * hidden size)


# model = vgg16()
# replace_conv(model, WSConv2d) # Original repo's implementation
# replace_conv(model, ScaledStdConv2d) # timm
# optim = torch.optim.SGD(model.parameters(), 1e-3) # Or any of your favourite optimizer
# optim = AGC(model.parameters(), optim)


class TorsoNetwork(nn.Module):
    '''
    this will be the torso network for both reward shaping module and RL policy model,
    this torso network will be pretrained by the replay observation video dataset
    we will append different head to complete the task
    (e.g., for video reconstruction, we append a transpose ConV2D head)
    '''

    def __init__(self, img_size, conv_depths, conv_dims, transformer_depths, transformer_dims, attn_head_num, drop_rate, cond_dim=None):
        super().__init__()
        self.conv_model = nn.Sequential(
            ConvNeXtMini(depths=conv_depths, dims=conv_dims, drop_path_rate=drop_rate)
        )
        self.img_size = img_size
        t = torch.rand((1, 3, img_size, img_size))
        t_out = self.conv_model(t)  # get -1 to get the final layer, we only use final layer at this moment
        self.after_conv_size = t_out.shape[-1]
        self.output_token_dim = transformer_dims
        conv_output_dim = t_out.numel()
        # add patch dropout
        self.conv_model.append(Rearrange('b c w h -> b (w h) c'))
        self.conv_model.append(nn.Dropout1d(p=drop_rate))
        self.conv_model.append(Rearrange('b wh c -> b (c wh)'))  # flatten size 256*8*8 = 16384
        if conv_output_dim != transformer_dims:
            conv_dim_convert_model = nn.Linear(conv_output_dim, transformer_dims)
            self.conv_model.append(conv_dim_convert_model)

        # according to Video diffusion model, kv can be the concatenation of video and sentence embedding -> shape [(video length + 1), feature_dim]
        self.transformer_model = nn.ModuleList([
            Gated_XAtten_Dense_Block(q_dim=transformer_dims,
                                     kv_dim=transformer_dims, depth=transformer_depths, num_heads=attn_head_num,
                                     drop_ratio=drop_rate, drop_path_ratio=drop_rate, attn_drop_ratio=drop_rate) for _
            in range(transformer_depths)
        ])

        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim * 4),
                nn.GELU(),
                nn.Linear(cond_dim * 4, transformer_dims)
            )
        else:
            self.cond_mlp = None

        # positional embedding
        self.rotatory_embed_generator = RotaryEmbedding(
            transformer_dims // attn_head_num)  # position embedding dim will be reduced by attn head
        self.pos_emb = self.rotatory_embed_generator(96)


    def forward(self, x, cond=None, mask=None):
        # conv
        # initial shape (batch, frame, channel, w, h)
        B = x.shape[0]
        L = x.shape[1]
        x = rearrange(x, 'b l c w h -> (b l) c w h')
        x = self.conv_model(x)
        # after conv, new shape is (big batch, high_dim channel,)
        x = rearrange(x, '(b l) D -> b l D', b=B)  # split
        # after flatten, the shape is (batch, frame, flatten_dim)
        if cond is not None and self.cond_mlp is not None:
            cond = self.cond_mlp(cond) # shape (batch, dim)
            cond = rearrange(cond, "b d -> b 1 d") # unsqueeze to match concat
            kv = torch.cat([cond, x], dim= -2)
        else:
            kv = x
        pos_emb = self.pos_emb[:kv.shape[-2]]

        # cross attention in transformer
        for block in self.transformer_model:
            x = block(x, kv, kv, mask, pos_emb)
        # lstm output's dim will be double the lstm dim
        return x



def test_model():
    net = TorsoNetwork(conv_depths=[3,3,9,3], conv_dims=[96,192,192,256], transformer_depths=3, transformer_dims=1024,
                       attn_head_num=8, drop_rate=0.2)
    net = net.to('cuda')
    input = torch.rand((2,15,3, 84, 84)).to('cuda:0') # (batch, frame, channel, w, h)
    output = net(input)
    print(output.shape) # should be (2, 15, 2048)

if __name__ == '__main__':
    test_model()