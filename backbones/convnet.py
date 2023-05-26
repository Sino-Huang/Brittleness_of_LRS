import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, repeat, rearrange
from torch import einsum
from einops.layers.torch import Reduce, Rearrange
from collections import OrderedDict
from nfnets import replace_conv, AGC, WSConv2d, ScaledStdConv2d, WSConvTranspose2d


import hydra
from omegaconf import DictConfig, OmegaConf

# ConvNet input shape: -> [batch, channel, H, W]

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


class LayerNorm(nn.Module): # layer norm that can apply to 2d conv
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(-3, keepdim=True)
            var = (x - mean).pow(2).mean(-3, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock(nn.Module):
    r""" ConvNeXtMini Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        # output size formula : [(W−K+2P)/S]+ 1
        # e.g., for W = 96, K = 7, P=3, S = 1, output_size = 96
        self.block_before_scale = nn.Sequential(OrderedDict([
            ('depthwise_conv', WSConv2d(dim, dim, kernel_size=7, padding=3, groups=dim)), # W, H remain the same, depthwise means we do convolution for each channel separately https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
            ('rearrange_1', Rearrange('b c h w -> b h w c')),
            ('layernorm', LayerNorm(dim, eps=1.0e-6, data_format="channels_last")),
            ('pointwise_conv_1', nn.Linear(dim, 4*dim)), # pointwise conv means 1x1 convs, we can use linear layer to implement it
            ('act', nn.GELU()),
            ('pointwise_conv_2', nn.Linear(4*dim, dim))
        ]))
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.block_before_scale(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(x)
        return x

# if patch size is 8, meaning one patch has 8 pixels,
# then for image size 96 * 96, we will have 12 * 12 number of patches

class ConvNeXtMini(nn.Module):
    r""" ConvNeXtMini
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 ):
        super().__init__()
        if depths is None:
            depths = [3,3]
        if dims is None:
            dims = [96, 192]
        # output size formula : [(W−K+2P)/S]+ 1
        # e.g., for W = 96, K = 4, P=0, S = 4, output_size = 24
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # input size 96, output size 24, input shape [B, C, H, W]
        stem = nn.Sequential(WSConv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # add downsampling layer
        for i in range(len(depths) - 1):
            #  for W = 24, K = 2, P=0, S = 2, output_size = 12
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             WSConv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvBlock(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (WSConv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = WSConv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)

class ConvSimpleDecoder(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.midddle_dim = in_dim // 4
        self.model = nn.Sequential(
            WSConvTranspose2d(in_dim, self.midddle_dim, 3, stride=1, padding=0),
            LayerNorm(self.midddle_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            WSConvTranspose2d(self.midddle_dim, self.midddle_dim, 5, stride=4, padding=0),
            LayerNorm(self.midddle_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            WSConvTranspose2d(self.midddle_dim, 3, 4, stride=5, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

def test_output_size():
    net = ConvNeXtMini(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    input = torch.rand((2,3, 84, 84))
    output = net(input)
    print(f'output shape is', output.shape)
    # output shape is torch.Size([2, 768, 2, 2])

    decodernet = ConvSimpleDecoder(in_dim=768)
    z = torch.rand((2,768, 2, 2))

    recon = decodernet(z) # output 0~ 1 need to change to 0 ~ 255
    print(f'recon shape is', recon.shape)



if __name__ == '__main__':
    test_output_size()