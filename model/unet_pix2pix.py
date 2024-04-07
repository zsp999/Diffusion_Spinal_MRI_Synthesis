import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#效果很好

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

# def zero_module(module):
#     """
#     Zero out the parameters of a module and return it.
#     """
#     for p in module.parameters():
#         p.detach().zero_()
#     return module

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    #改成nn.InstanceNorm2d nn.InstanceNorm2d(channels)
    return  nn.InstanceNorm2d(channels) #GroupNorm32(16, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Downsample(nn.Module): #卷积降维或者无卷积池化（不改通道数） 无激活
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2) #降维
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module): #上采样+卷积
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv #一般是false
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), #如果特征图变成1@1没有这一步
            nn.ReLU() if up else nn.LeakyReLU() if down else nn.SiLU(),
            # nn.SiLU(),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.in_layers_onepix = nn.ReLU() if up else nn.LeakyReLU() if down else nn.SiLU()

        self.updown = up or down #升维或降维都是True

        if up:
            self.h_upd = Upsample(channels, self.use_conv, dims)
            self.x_upd = Upsample(channels, self.use_conv, dims)
        elif down:
            self.h_upd = Downsample(channels, self.use_conv, dims)
            self.x_upd = Downsample(channels, self.use_conv, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()


        # self.out_layers = nn.Sequential(
        #     normalization(self.out_channels),
        #     nn.ReLU() if up else nn.LeakyReLU() if down else nn.SiLU(), #nn.SiLU()
        #     nn.Dropout(p=dropout),
        #     conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),

        # )
        self.out_layers =conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x): #不要time emb
        self.in_layers = self.in_layers_onepix if x.shape[-1]==1 else self.in_layers
        if self.updown:
            # in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.in_layers(x)#norm+激活
            h = self.h_upd(h) #变维
            h = self.out_layers(h)
            # h = in_conv(h)

            x = self.x_upd(x) #直接升维 再经过卷积（或者直接输出）
             
        else:
            h = self.out_layers(self.in_layers(x))
        # h = self.out_layers(h) #减少一下参数
        return self.skip_connection(x) + h



# https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/openaimodel.py#L413
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        # attention_resolutions,
        use_conv = False, #改成True试试
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=False, #
        dims=2, # 2D卷积
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False, #用FP32
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        # use_new_attention_order=False, #false还是true
        use_spatial_transformer=False, #关闭   # custom transformer support
        legacy=True,
        use_tanh = True
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        # self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.input_blocks = nn.ModuleList(
            [
            conv_nd(dims, in_channels, model_channels, 3, padding=1)
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels #ch是初始channel
        ds = 1
        for level, mult in enumerate(channel_mult): #level0 1 2 3，mult1 2 4 8
            for _ in range(num_res_blocks): #几次resnet:1
                layers = [
                    ResBlock(
                        ch,
                        # time_embed_dim,
                        dropout,
                        use_conv = use_conv,
                        out_channels=mult * model_channels, #输出mutl * 初始通道
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels #ch更换为mutl * 初始通道

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1: #如果不是最后一个，如4
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            # time_embed_dim,
                            dropout,
                            use_conv = use_conv,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch #更改ch为上一次输出
                input_block_chans.append(ch)
                ds *= 2 #扩大两倍ds
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                # time_embed_dim,
                dropout,
                dims=dims,
                use_conv = use_conv,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            #attn一点不要！！！！
            ResBlock(
                ch,
                # time_embed_dim,
                dropout,
                dims=dims,
                use_conv = use_conv,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: #8,4,2,1
            for i in range(num_res_blocks + 1): #每一次升维包括num_res_blocks+1个resnet
                ich = input_block_chans.pop() #把之前encoder的通道加进去
                layers = [
                    ResBlock(
                        ch + ich,
                        # time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        use_conv = use_conv,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                #attn不要
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            # time_embed_dim,
                            dropout,
                            use_conv = use_conv,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2 #ds每次缩小两倍
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.ReLU(),
            # nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1), #zero_module(
            nn.Tanh() if use_tanh else nn.Identity(), #加上这个试试 需要！！！
        )


    def forward(self, x):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h) #, emb, context
            hs.append(h)
        h = self.middle_block(h) #, emb, context
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h) #, emb, context
        h = h.type(x.dtype)
        return self.out(h)
    


#patchGAN 鉴别器: 小的效果好 
class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleDiscriminator, self).__init__()

        def discriminator_block_downsize(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.InstanceNorm2d(out_filters), nn.LeakyReLU(0.2, inplace=True)]
            return layers
        
        def discriminator_block_nodownsize(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1), nn.InstanceNorm2d(out_filters), nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block_downsize(in_channels, 64), #256--128
            *discriminator_block_downsize(64, 128), #128--64
            *discriminator_block_downsize(128, 256), #64--32
            *discriminator_block_nodownsize(256, 512), #32--32 
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B): #(bz, channel, 256, 256)
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


