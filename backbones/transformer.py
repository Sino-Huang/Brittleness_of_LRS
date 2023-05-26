from random import shuffle, seed, randint
import math
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from einops import reduce, repeat, rearrange

# Transformer input shape: -> [Batch, Sequence_len, Feature_dim] for q, k, v
# ---------------------- Rotatary Embedding ----------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim): # dimension of the input, divided by attention head though
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq) # register_buffer means we will store this value when we store pytorch model

    def forward(self, max_seq_len, *, device='cpu'):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq.to(device))
        return torch.cat((freqs, freqs), dim=-1)


def apply_rotary_pos_emb(pos, t):
    pos = pos.to(t.device)
    def rotate_half(x):
        x = rearrange(x, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    return (t * pos.cos()) + (rotate_half(t) * pos.sin())
# ---------------------- end Rotatary Embedding ----------------------------

class SwiGLU(nn.Module):
    # note that this will divide the feature dim by 2
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def exists(val):
    return val is not None

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        *,
        eps = 1e-8,
        gated = False
    ):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim)) if gated else None

    def forward(self, x):
        norm = x.norm(keepdim = True, dim = -1) * self.scale
        out = (x / norm.clamp(min = self.eps)) * self.gamma

        if not exists(self.weight):
            return out

        return out * (x * self.weight).sigmoid()

class LayerNorm(nn.Module):
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
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


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


class NormMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, factor = 4, dropout_rate = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor
        self.middle_dim = int(self.factor * self.input_dim)
        self.block = nn.Sequential(
            nn.Linear(self.input_dim, self.middle_dim * 2, bias=False),
            SwiGLU(),
            nn.Dropout(dropout_rate),
            RMSNorm(self.middle_dim),
            nn.Linear(self.middle_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 ):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x, mask= None, pos_emb = None): # mask 1 means we keep it during attention, 0 means we do not use it
        # [batch_size, num_patches + 1, total_embed_dim] # + 1 means we are having a cls token for prediction
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # scale
        q = q * self.scale

        # add positional embedding for each attn layer
        if pos_emb is not None:
            pos_emb = pos_emb.type(q.dtype)
            q = apply_rotary_pos_emb(pos_emb[:q.shape[-2]], q)
            k = apply_rotary_pos_emb(pos_emb[:k.shape[-2]], k)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1))
        # masking
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j') # expand to get correct dim
            attn = attn.masked_fill(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    r'''
    we need to consider the case where q query shape is not the same as k and v. we need to ensure that the output shape is the same as q
    '''
    def __init__(self,
                 q_dim,   # 输入token的dim
                 kv_dim,
                 v_dim = None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 ):
        r'''
        :param q_dim: usually the target dim
        :param kv_dim: original action and image dim, should be the same dim
        :param num_heads: multi head attention
        :param qkv_bias: whether need bias param
        :param qk_scale:
        :param attn_drop_ratio: attention drop path rate
        :param proj_drop_ratio: linear dropout rate
        '''
        super(CrossAttention, self).__init__()
        if v_dim is None:
            v_dim = kv_dim
        self.num_heads = num_heads
        head_dim = q_dim // num_heads # our final output shape should be target_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.qW = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kW = nn.Linear(kv_dim, q_dim, bias=qkv_bias)
        self.vW = nn.Linear(v_dim, q_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(q_dim, q_dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, q, k, v, mask=None, pos_emb=None, k_pos_only=False):
        # [batch_size, seq_length, total_embed_dim]
        B, N, C= q.shape # N is the seq_length
        kv_N = k.shape[1]

        # q(): -> [batch_size, q_seq_length, total_embed_dim]
        # k(),v(): -> [batch_size, kv_seq_length, total_embed_dim]
        q = self.qW(q)
        k = self.kW(k)
        v = self.vW(v)
        # split head [batch_size, num_heads, seq_length, embed_dim_per_head]
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))
        # scale
        q = q * self.scale

        # add positional embedding for each attn layer
        if pos_emb is not None:
            pos_emb = pos_emb.type(q.dtype)
            if not k_pos_only:
                q = apply_rotary_pos_emb(pos_emb[:q.shape[-2]], q)
            k = apply_rotary_pos_emb(pos_emb[:k.shape[-2]], k)
        # k_transpose: -> [batch_size, num_heads, embed_dim_per_head, kv_seq_length]
        # @: multiply -> [batch_size, num_heads, q_seq_length, kv_seq_length]
        attn = (q @ k.transpose(-2, -1))

        # masking
        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j') # expand to get correct dim
            attn = attn.masked_fill(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # v: -> [batch_size, num_heads, v_seq_length, embed_dim_per_head]
        # @: multiply -> [batch_size, num_heads, q_seq_length, embed_dim_per_head]
        # transpose: -> [batch_size, q_seq_length, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, q_seq_length, total_embed_dim] # the shape is the same as the shape of input q
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Gated_XAtten_Dense_Block(nn.Module):
    def __init__(self,
                 q_dim,
                 kv_dim,
                 depth,
                 v_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.2,
                 attn_drop_ratio=0.2,
                 drop_path_ratio=0.2,  # set dropout rate to be 0.2 as default
                 act_layer=SwiGLU,
                 norm_layer=LayerNorm):
        r'''
        :param q_dim:  q feature dim, it should be the target encoding feature size, please change lang feature size to target size in advance
        :param kv_dim: orignial k and v feature dim, should be the same size, (action and image)
        :param num_heads: num of heads for multi head attention
        :param qkv_bias: whether need bias param
        :param qk_scale:
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: droprate in the attention
        :param drop_path_ratio: droppath rate between layers
        :param act_layer:
        :param norm_layer:
        '''

        super(Gated_XAtten_Dense_Block, self).__init__()
        if v_dim is None:
            v_dim = kv_dim
        self.norm1_q = norm_layer(q_dim)
        self.norm1_k = norm_layer(kv_dim)
        self.norm1_v = norm_layer(v_dim)
        self.norm1_post = norm_layer(q_dim)
        self.crossattn = CrossAttention(q_dim=q_dim,kv_dim=kv_dim, v_dim=v_dim,  num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.selfattn_option_freeze_pretrain = SelfAttention(dim=q_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                             qk_scale=qk_scale,
                                                             attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        self.norm3 = norm_layer(q_dim)
        self.norm3_post = norm_layer(q_dim)
        self.norm4 = norm_layer(q_dim)
        self.mlp1 = NormMLPBlock(input_dim=q_dim, output_dim=q_dim, dropout_rate=drop_ratio)
        self.mlp2_option_freeze_pretrain = NormMLPBlock(input_dim=q_dim, output_dim=q_dim, dropout_rate=drop_ratio)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.mlp_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, q, k, v, mask=None, pos_emb=None, cross_pos_only=False, k_pos_only=False ):
        q = q + self.drop_path(self.norm1_post(self.crossattn(self.norm1_q(q), self.norm1_k(k), self.norm1_v(v), mask, pos_emb, k_pos_only))) * self.attn_gate.tanh()
        q = q + self.drop_path(self.mlp1(self.norm2(q))) * self.mlp_gate.tanh()
        q = q + self.drop_path(self.norm3_post(self.selfattn_option_freeze_pretrain(self.norm3(q), mask, None if cross_pos_only else pos_emb)))
        q = q + self.drop_path(self.mlp2_option_freeze_pretrain(self.norm4(q)))
        return q