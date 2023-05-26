import random
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from backbones import Gated_XAtten_Dense_Block, RotaryEmbedding, Bottle2neck, Res2Net, RMSNorm
from backbones import TorsoNetwork
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


def drop_patch(x, drop_prob: float = 0., training: bool = False):
    # x shape [b, c, w, h]
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) + (x.shape[2], x.shape[3])  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPatch(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPatch, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # shape x [b, c, w, h]
        return drop_patch(x, self.drop_prob, self.training)

class SimpleImgTransDecoder(nn.Module):
    def __init__(self, dim, IMG_SIZE):
        super(SimpleImgTransDecoder, self).__init__()

        self.decoder_norm = RMSNorm(dim)
        # after this encoder decoder the data shape is [B, L+1, Feature_dim]
        # we need to change back to [B, 3, image_size, image_size]
        # so we need to ensure L * output_dim = 3 * image_size**2
        self.decoder_pred_img = nn.Linear(dim, 3 * IMG_SIZE ** 2, bias=True)
        self.IMG_SIZE = IMG_SIZE

    def forward(self, x):
        # shape (B, L, dim)
        x = self.decoder_norm(x)
        x = self.decoder_pred_img(x)
        x = rearrange(x, "b l (c w h) -> b l c w h", c= 3, w=self.IMG_SIZE, h=self.IMG_SIZE)
        x = torch.nn.functional.sigmoid(x) # range from 0 to 1
        return x

class DeprecatedTorsoNetwork(nn.Module):
    '''
    this will be the torso network for both reward shaping module and RL policy model,
    this torso network will be pretrained by the replay observation video dataset
    we will append different head to complete the task
    (e.g., for video reconstruction, we append a transpose ConV2D head)
    '''

    def __init__(self, img_size, conv_depths, transformer_depths, transformer_dims, attn_head_num, lstm_layer_num,
                 lstm_dims, drop_rate, cond_dim=None):
        super().__init__()
        self.conv_model = nn.Sequential(
            Res2Net(Bottle2neck, conv_depths, baseWidth=26, scale=4)
        )
        self.img_size = img_size
        t = torch.rand((1, 3, img_size, img_size))
        t_out = self.conv_model(t) # get -1 to get the final layer, we only use final layer at this moment
        self.after_conv_size = t_out.shape[-1]
        self.output_token_dim = 2 * lstm_dims
        conv_output_dim = t_out.numel()
        if conv_output_dim != transformer_dims:
            conv_dim_convert_model = nn.Linear(conv_output_dim, transformer_dims)
            # add patch dropout
            self.conv_model.append(Rearrange('b c w h -> b (w h) c'))
            self.conv_model.append(nn.Dropout1d(p=drop_rate))
            self.conv_model.append(Rearrange('b wh c -> b (c wh)')) # flatten size 256*8*8 = 16384
            self.conv_model.append(conv_dim_convert_model)

        # self.conv_model = torch.jit.trace(self.conv_model, example_inputs=t, check_inputs=[(torch.rand((2, 3, img_size, img_size)),)])

        # positional embedding
        self.rotatory_embed_generator = RotaryEmbedding(
            transformer_dims // attn_head_num)  # position embedding dim will be reduced by attn head
        self.pos_emb = self.rotatory_embed_generator(30)

        # according to Video diffusion model, kv can be the concatenation of video and sentence embedding -> shape [(video length + 1), feature_dim]
        # t= torch.rand((1,2,transformer_dims))
        droppathrate = [x.item() for x in torch.linspace(0, drop_rate,
                                                         transformer_depths)]  # stochastic depth decay
        self.transformer_model = nn.ModuleList([
            Gated_XAtten_Dense_Block(q_dim=transformer_dims,
                                     kv_dim=transformer_dims, depth=transformer_depths, num_heads=attn_head_num,
                                     drop_ratio=drop_rate, drop_path_ratio=droppathrate[ind], attn_drop_ratio=droppathrate[ind]) for ind
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

        self.lstm_model = nn.GRU(transformer_dims, lstm_dims, num_layers=lstm_layer_num, batch_first=True, bidirectional=True)




    def forward(self, x, cond=None, mask=None):
        # conv
        # initial shape (batch, frame, channel, w, h)
        B = x.shape[0]
        x = rearrange(x, 'b l c w h -> (b l) c w h')
        x = self.conv_model(x)
        # after conv, new shape is (big batch, high_dim channel, w, h)
        x = rearrange(x, '(b l) D -> b l D', b=B) # split

        # after flatten, the shape is (batch, frame, flatten_dim)
        if cond is not None and self.cond_mlp is not None:
            cond = self.cond_mlp(cond)  # shape (batch, dim)
            cond = rearrange(cond, "b d -> b 1 d")  # unsqueeze to match concat
            kv = torch.cat([cond, x], dim= -2)
        else:
            kv = x

        pos_emb = self.pos_emb[:kv.shape[-2]]

        # cross attention in transformer
        for block in self.transformer_model:
            x = block(x, kv, kv, mask, pos_emb)
        # lstm output's dim will be double the lstm dim
        # output shape (batch, frame, 2x lstm dims)
        x,_ = self.lstm_model(x)
        return x # return value and info dict

class Video_Autoencoder(nn.Module):
    def __init__(self, rl_img_size, *args, **kwargs):
        super().__init__()
        # self.torsonet = DeprecatedTorsoNetwork(*args, **kwargs)
        self.torsonet = TorsoNetwork(*args, **kwargs, conv_dims= [96,192,256])
        self.after_conv_size = self.torsonet.after_conv_size
        self.output_token_dim = self.torsonet.output_token_dim
        print('after_conv_size', self.after_conv_size)
        self.rl_img_size = rl_img_size
        self.decoder = SimpleImgTransDecoder(self.output_token_dim, self.rl_img_size)

    def forward(self, x):
        #init shape x (batch, frame, channel, W, H)
        B = x.shape[0]
        x = self.torsonet(x)
        # after torsonet, shape is (batch, frame, large dim )
        x = self.decoder(x) # output range from 0 to 1
        return x


    @staticmethod
    def handle_data(x, WINDOW_SIZE, CLIP_LENGTH):
        output = x[0]["video"]
        output_resize = x[0]["video_target"]
        return [output, output_resize]


    def save(self, path):
        r'''
        :param path: path to save model
        :return:
        '''
        torch.save(self.state_dict(), path)

def test_model():

    # net = Video_Autoencoder(rl_img_size = 84, img_size=256, conv_depths=[3, 8, 16, 3], transformer_depths=1, transformer_dims=1024,
    #                    attn_head_num=8, lstm_layer_num=1, lstm_dims=512, drop_rate=0.2)
    net = Video_Autoencoder(
        rl_img_size=84,
        img_size=84,
        conv_depths=[3, 16, 9],
        transformer_depths=1,
        transformer_dims=2048,
        attn_head_num=8,
        drop_rate=0.2,
    )
    input = torch.rand((6,15,3, 84, 84))# (batch, frame, channel, w, h)

    # net = torch.jit.script(net, input)
    net = net.to('cuda')
    input = input.to("cuda")
    output = net(input)
    print(output.shape)
    # net.save('temp.pt')


if __name__ == '__main__':
    test_model()