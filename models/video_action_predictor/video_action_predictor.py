import random

import torch
import torch.nn as nn
import torch.functional as F
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


class ImgDiffNetwork(nn.Module):
    '''
    get a sequence of img, calculate the img_t - img_t-1, apply Conv2D and flatten
    '''

    def __init__(self, conv_depths, conv_dims, drop_rate = 0.1): # conv_depths=[3,3], conv_dims= [96,96]
        super().__init__()
        self.conv_model = ConvNeXtMini(in_chans=1, depths=conv_depths, dims=conv_dims, drop_path_rate=drop_rate)

    @staticmethod
    def preprocess_images(x): # x shape (batch, frame, c, w, h)
        # output batch, frame-1 w, h
        # grayscale
        B = x.shape[0]
        x = 0.299 * x[:,:,0] + 0.587 * x[:,:,1] + 0.114 * x[:,:,2]
        # calculate diff
        x = x[:,1:] - x[:,:-1] # shape (b, f-1, w, h)
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        return x, B # x shape (Big Batch, 1, w, h) , B = batch

    def forward(self, x): # x shape (batch, w, h)
        x = self.conv_model(x)
        # flatten
        x = rearrange(x, "b c w h -> b (c w h)")
        return x

class Video_Action_Predictor(nn.Module):
    def __init__(self, ACTION_SIZE, device, img_size, conv_depths, conv_dims):
        super().__init__()
        self.device = device
        self.torsonet = ImgDiffNetwork(conv_depths, conv_dims)
        t = torch.rand((1, 1, img_size, img_size))  # shape (batch, 1, w, h)
        t_out = self.torsonet(t)  # shape (batch, frame, lstm_dim)
        dim = t_out.shape[-1]
        self.head = nn.Linear(dim, ACTION_SIZE)

    def forward(self, x):
        # init shape x (batch, LEN(2), channel, W, H)
        x = self.torsonet(x)
        # after torsonet, shape is (batch, frame, large dim )
        x = self.head(x) # only select the last frame
        return x

    def handle_data(self, x):
        img_pair = x[0]['img_diff']
        action = x[0]['action']
        img_pair = img_pair.to(self.device, non_blocking = True)
        action = action.to(self.device, non_blocking = True)
        return img_pair, action

    def save(self, path):
        r'''
        :param path: path to save model
        :return:
        '''
        torch.save(self.state_dict(), path)


def test_model():
    net = Video_Action_Predictor(ACTION_SIZE=8, device="cuda:0", img_size=84,   conv_depths= [3,3,3], conv_dims =[96,96,96])
    net = net.to('cuda')
    input = torch.rand((2, 1, 84, 84)).to('cuda:0')  # (batch, 1, w, h)
    output = net(input)
    print(output.shape)


if __name__ == '__main__':
    test_model()
