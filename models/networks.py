from os.path import isfile

import torch
import torch.nn as nn
import functools, itertools
import numpy as np
from einops import rearrange, repeat
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from kornia.color import rgb_to_lab, lab_to_rgb
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import resnet18, ResNet18_Weights

from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import SSIM
from models.utils_fct import get_norm_layer, weights_init, power_iteration
from util.util import gkern_2d
import torch.nn.functional as F
from torchvision import models
import math


# Convenience passthrough function
class identity(nn.Module):
    def forward(self, input):
        return input


class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


################################SN#######################

#######Positional Encoding module, borrowed from https://github.com/open-mmlab/mmgeneration
class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]

        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class CatersianGrid(nn.Module):
    """Catersian Grid for 2d tensor.
    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)


#######################################Positional Encoding############

##########Central Difference Convolution, borrowed from https://github.com/ZitongYu/CDCN/
class CDC2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x[:, :, 1:-1, 1:-1], weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)
            # print(out_normal.size())
            # print(out_diff.size())
            # print(x.size())

            return out_normal - self.theta * out_diff


###############

def define_G(input_nc, output_nc, ngf, net_Gen_type, n_blocks, n_blocks_shared, n_domains, norm='batch',
             use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if net_Gen_type == 'gen_v1':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoderv1, dec_args)
    elif net_Gen_type == 'gen_SGPA':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoderv2, enc_args, ResnetGenDecoderv1, dec_args)
    else:
        raise NotImplementedError('Generation Net [%s] is not found' % net_Gen_type)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminatorSN, model_args)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD


def define_S(input_nc, ngf, n_blocks, n_domains, num_classes=19, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, n_blocks, ngf, num_classes, norm_layer, use_dropout, gpu_ids)
    plex_netS = S_Plexer(n_domains, SegmentorHeadv2, model_args)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        plex_netS.cuda(gpu_ids[0])

    plex_netS.apply(weights_init)
    return plex_netS


class Vgg16(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features.cuda(gpu_ids)
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        # for x in range(23, 30):
        #     self.to_relu_5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # h = self.to_relu_5_3(h)
        # h_relu_5_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class Get_gradmag_gray(nn.Module):
    "To obtain the magnitude values of the gradients at each position."

    def __init__(self):
        super(Get_gradmag_gray, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).to(torch.device(0))
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda(torch.device(0))

    def forward(self, x):
        x_norm = (x + 1) / 2
        x_norm = (.299 * x_norm[:, :1, :, :] + .587 * x_norm[:, 1:2, :, :] + .114 * x_norm[:, 2:, :, :])
        x0_v = F.conv2d(x_norm, self.weight_v, padding=1)
        x0_h = F.conv2d(x_norm, self.weight_h, padding=1)

        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x_gradmagn


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


###Add 1 Pyramid Guided Attention Block v4 before ResBlock groups
class ResnetGenEncoderv2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenEncoderv2, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.module_SGPA = SGPABlock(ngf * mult, norm_layer=norm_layer, use_bias=use_bias, gpu_ids=gpu_ids,
                                     padding_type='reflect')

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            temp_fea = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            _, _, h, w = temp_fea.size()
            input_resize = F.interpolate(input, size=(h, w), mode='bilinear', align_corners=False)
            out, attmap1, attmap2, attmap3 = nn.parallel.data_parallel(self.module_SGPA,
                                                                       torch.cat((temp_fea, input_resize), 1),
                                                                       self.gpu_ids)

        else:
            temp_fea = self.model(input)
            _, _, h, w = temp_fea.size()
            input_resize = F.interpolate(input, size=(h, w), mode='bilinear', align_corners=False)
            out, attmap1, attmap2, attmap3 = self.module_SGPA(torch.cat((temp_fea, input_resize), 1))

        return out, attmap1, attmap2, attmap3


####Add 1 Pyramid Guided Attention Block v4 before ResBlock groups
class ResnetGenDecoderv1(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenDecoderv1, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(32, int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)


class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


# Define a SGPABlock, attention maps of three scale are fused adaptively. Spatial Gradient Pyramid Attention Module
class SGPABlock(nn.Module):
    def __init__(self, in_dim, norm_layer, use_bias, gpu_ids=[], padding_type='reflect'):
        super(SGPABlock, self).__init__()

        self.gpu_ids = gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        # self.dsamp_filter = self.Tensor([1]).view(1,1,1,1)
        self.grad_filter = self.Tensor([0, 0, 0, -1, 0, 1, 0, 0, 0]).view(1, 1, 3, 3)

        self.GradConv = nn.Sequential(nn.Conv2d(2, 32, kernel_size=7, padding=3, bias=use_bias, padding_mode='zeros'),
                                      norm_layer(32), nn.PReLU())
        self.GradAtt = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=use_bias, padding_mode='zeros'),
                                     norm_layer(32), nn.PReLU(),
                                     nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=use_bias, padding_mode='zeros'),
                                     nn.Sigmoid())

        self.ConvLK1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type),
            norm_layer(in_dim), nn.PReLU())
        self.ConvCF1 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32),
                                     nn.PReLU())
        self.ConvLK2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type),
            norm_layer(in_dim), nn.PReLU())
        self.ConvCF2 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32),
                                     nn.PReLU())
        self.ConvLK3 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type),
            norm_layer(in_dim), nn.PReLU())
        self.ConvCF3 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32),
                                     nn.PReLU())

        self.ds1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), nn.AvgPool2d(kernel_size=2, stride=2))
        self.ConvCF_Up1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type),
            norm_layer(in_dim), nn.PReLU())
        self.ConvCF_Up2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type),
            norm_layer(in_dim), nn.PReLU())

    def getgradmaps(self, input_gray_img):
        dx = F.conv2d(input_gray_img, self.grad_filter, padding=1)
        dy = F.conv2d(input_gray_img, self.grad_filter.transpose(-2, -1), padding=1)
        gradient = torch.cat([dx, dy], 1)
        # x_gradmagn = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + 1e-6)

        return gradient

    def forward(self, x):
        _, _, h, w = x.size()
        # print(x.size())
        input_fea = x[:, :-3, :, :]
        ori_img = x[:, -3:, :, :]
        gray = (.299 * ori_img[:, 0, :, :] + .587 * ori_img[:, 1, :, :] + .114 * ori_img[:, 2, :, :]).unsqueeze_(1)

        gray_dsamp1 = F.interpolate(gray, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        gray_dsamp2 = F.interpolate(gray, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        gray_dsamp3 = F.interpolate(gray, size=(h, w), mode='bilinear', align_corners=False)

        gradfea1 = self.GradConv(self.getgradmaps(gray_dsamp1))
        gradfea2 = self.GradConv(self.getgradmaps(gray_dsamp2))
        gradfea3 = self.GradConv(self.getgradmaps(gray_dsamp3))

        fea_ds1 = self.ds1(input_fea)
        fea_LKC1 = self.ConvLK1(fea_ds1)
        fea_CF1 = self.ConvCF1(fea_LKC1)
        gradattmap1 = self.GradAtt(torch.cat([fea_CF1, gradfea1], 1))
        fea_att1 = gradattmap1.expand_as(fea_LKC1).mul(fea_LKC1) + fea_ds1

        fea_ds2 = self.ConvCF_Up1(F.interpolate(fea_att1, size=(h // 2, w // 2), mode='bilinear', align_corners=False))
        fea_LKC2 = self.ConvLK2(fea_ds2)
        fea_CF2 = self.ConvCF2(fea_LKC2)
        gradattmap2 = self.GradAtt(torch.cat([fea_CF2, gradfea2], 1))
        fea_att2 = gradattmap2.expand_as(fea_LKC2).mul(fea_LKC2) + fea_ds2

        fea_ds3 = self.ConvCF_Up2(F.interpolate(fea_att2, size=(h, w), mode='bilinear', align_corners=False))
        fea_LKC3 = self.ConvLK3(fea_ds3)
        fea_CF3 = self.ConvCF3(fea_LKC3)
        gradattmap3 = self.GradAtt(torch.cat([fea_CF3, gradfea3], 1))
        out = gradattmap3.expand_as(fea_LKC3).mul(fea_LKC3) + input_fea

        AM1_us = F.interpolate(gradattmap1, size=(h, w), mode='bilinear', align_corners=False)
        AM2_us = F.interpolate(gradattmap2, size=(h, w), mode='bilinear', align_corners=False)

        return out, AM1_us, AM2_us, gradattmap3


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.PReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_dim, num_classes, norm_layer):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(in_dim, 64, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(in_dim, 64, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(in_dim, 64, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(in_dim, 64, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_dim, 64, 1, stride=1, bias=False),
                                             nn.PReLU())
        self.conv1 = nn.Conv2d(320, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv_1x1_4(self.dropout(x))

        return out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0, act=None):
        super(ResnetBlock, self).__init__()
        act = act if act is not None else nn.PReLU()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)


class ResnetBlock2(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0, act=None):
        super(ResnetBlock2, self).__init__()
        act = act if act is not None else nn.PReLU()
        self.act = act
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       act]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)
        self.fus_conv = nn.Conv2d(2 * dim, 4 * dim, kernel_size=3, padding=1, bias=use_bias)
        self.ssim = StructuralSimilarityIndexMeasure(gaussian_kernel=True,
                                                                 sigma=1.5,
                                                                 kernel_size=11,
                                                                 reduction=None,
                                                                 data_range=None,
                                                                 k1=0.01, k2=0.03,
                                                                 return_full_image=True,
                                                                 return_contrast_sensitivity=False).to('cuda')
        self.final_block = nn.Sequential(nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1, bias=use_bias),
                                         norm_layer(dim),
                                         nn.PReLU())


    def forward(self, inp, *args):
        mask = args[0] if args else None
        if isinstance(inp, tuple):
            x, y = inp
            b, c, h, w = x.shape
            x_ = x
            y_ = self.conv_block(y)
            y_ = self.filter(y_, mask) if mask is not None else y_
            xy_ = torch.cat([x_, y_], dim=1)
            res = self.fus_conv(xy_)
            x_, y_ = res.split(2*c, 1)
            mask = self.compute_mask(x_, y_)
            res = mask * x_ + (1-mask) * y_
            # res = torch.max(torch.cat([x_.reshape(1, -1), y_.reshape(1, -1)]), dim=0)[0]
            return self.final_block(res)
        return inp + self.conv_block(inp)

    def filter(self, feat, mask):
        mask = mask.squeeze()
        b, c_f, h_f, w_f = feat.shape
        filter_r = repeat(mask, 'h w -> b c_f h w', b=b, c_f=c_f)
        filter_s = F.interpolate(filter_r, feat.shape[-2:])
        return feat * filter_s

    def compute_mask(self, feat_x, feat_y):
        im_x = ImageTensor(feat_x * 0.5 + 1)
        im_y = ImageTensor(feat_y * 0.5 + 1)
        _, mask = self.ssim(im_x, im_y)
        self.ssim.reset()
        return torch.abs(mask.detach())


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, weights=None, act=None, norm=None, **kwargs):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        if weights is not None:
            assert weights.shape == (out_features, in_features), "Weights shape mismatch"
            self.weight = nn.Parameter(weights)
        self.norm = norm(out_features)
        self.act = act

    def forward(self, inp):
        inp = inp.permute(0, 2, 3, 1).contiguous()  # Change to (N, H, W, C)
        output = super(Linear, self).forward(inp)
        output = output.permute(0, 3, 1, 2).contiguous()  # Change back to (N, C, H, W)
        if self.norm is not None:
            output = self.norm(output)
        if self.act is not None:
            output = self.act(output)
        return output


class Conv2d(nn.Conv2d):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1,
                 bias=True, weights=None, act=None, norm=None, stride=1, **kwargs):
        super(Conv2d, self).__init__(in_features, out_features, kernel_size=kernel_size, padding=padding, stride=stride)
        if weights is not None:
            assert weights.shape == (out_features, in_features, kernel_size, kernel_size), "Weights shape mismatch"
            self.weight = nn.Parameter(weights)
        self.norm = norm(out_features)
        self.act = act

    def forward(self, inp):
        output = super(Conv2d, self).forward(inp)
        if self.norm is not None:
            output = self.norm(output)
        if self.act is not None:
            output = self.act(output)
        return output


class ConcatBlock(nn.Module):
    def __init__(self, dim, nb_blocks, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0,
                 gpu_ids=[], ratio=0.1):
        super(ConcatBlock, self).__init__()
        self.ratio = ratio
        self.gpu_ids = gpu_ids
        act = nn.Tanh()
        conv_block_fus = []
        conv_block_sep = []

        self.dim = dim
        self.n_domains = n_domains
        self.use_bias = use_bias
        self.act = act
        self.norm_layer = norm_layer
        if use_dropout:
            conv_block_fus += [nn.Dropout(0.5)]

        conv_block_fus += [ResnetBlock2(dim, norm_layer, use_dropout, use_bias, padding_type, n_domains,
                                        act)] * nb_blocks
        #
        self.conv_block_fus = nn.Sequential(*conv_block_fus)

    def forward(self, x_input, y_input=None, *args):
        mask = args[0] if args else None
        z = y_input
        for conv in self.conv_block_fus:
            z = conv((x_input, z), mask)
        return z


class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', **kwargs):
        super(ResBlock, self).__init__()
        p = 0
        conv_block = []
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(2)]
        elif padding_type == 'zero':
            p = 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=5, padding=p, bias=use_bias), norm_layer(dim), nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        self.conv_block = nn.Sequential(*conv_block)
        self.norm_layer = norm_layer(dim)

    def forward(self, x):
        return self.norm_layer(x + self.conv_block(x))


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0, 0, 0, -1, 0, 1, 0, 0, 0]).view(1, 1, 3, 3)
        self.dsamp_filter = tensor([1]).view(1, 1, 1, 1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers - 1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(), \
 \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299 * input[:, 0, :, :] + .587 * input[:, 1, :, :] + .114 * input[:, 2, :, :]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2, -1))
        gradient = torch.cat([dx, dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3


EPS = 1e-6


class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
    Args:
      h: histogram dimension size (scalar). The default value is 64.
      insz: maximum size of the input image; if it is larger than this size, the
        image will be resized (scalar). Default value is 150 (i.e., 150 x 150
        pixels).
      resizing: resizing method if applicable. Options are: 'interpolation' or
        'sampling'. Default is 'interpolation'.
      method: the method used to count the number of pixels for each bin in the
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
      sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
        the sigma parameter of the kernel function. The default value is 0.02.
      intensity_scale: boolean variable to use the intensity scale (I_y in
        Equation 2). Default value is True.

    Methods:
      forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
         training.
    """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if self.method == 'thresholding':
            self.eps = 6.0 / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h)).to(
            device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS), dim=1)
            else:
                Iy = 1

            Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS), dim=1)
            Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS), dim=1)
            diff_u0 = abs(Iu0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)), dim=0).to(self.device))
            diff_v0 = abs(Iv0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)), dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)), 2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)), 2) / self.sigma ** 2
                diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                diff_v0 = torch.exp(-diff_v0)
            elif self.method == 'inverse-quadratic':
                diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)), 2) / self.sigma ** 2
                diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)), 2) / self.sigma ** 2
                diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                diff_v0 = 1 / (1 + diff_v0)
            else:
                raise Exception(
                    f'Wrong kernel method. It should be either thresholding, RBF,'
                    f' inverse-quadratic. But the given value is {self.method}.')
            diff_u0 = diff_u0.type(torch.float32)
            diff_v0 = diff_v0.type(torch.float32)
            a = torch.t(Iy * diff_u0)
            hists[l, 0, :, :] = torch.mm(a, diff_v0)

            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS), dim=1)
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS), dim=1)
            diff_u1 = abs(Iu1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)), dim=0).to(self.device))
            diff_v1 = abs(Iv1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)), dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            hists[l, 1, :, :] = torch.mm(a, diff_v1)

            Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1)
            Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS),
                                  dim=1)
            diff_u2 = abs(
                Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            diff_v2 = abs(
                Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                                      dim=0).to(self.device))
            if self.method == 'thresholding':
                diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = torch.exp(-diff_u2)  # Gaussian
                diff_v2 = torch.exp(-diff_v2)
            elif self.method == 'inverse-quadratic':
                diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                diff_v2 = 1 / (1 + diff_v2)
            diff_u2 = diff_u2.type(torch.float32)
            diff_v2 = diff_v2.type(torch.float32)
            a = torch.t(Iy * diff_u2)
            hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminatorSN, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0, 0, 0, -1, 0, 1, 0, 0, 0]).view(1, 1, 3, 3)
        self.dsamp_filter = tensor([1]).view(1, 1, 1, 1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers - 1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequences = [[
            SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequences += [[
                SNConv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                         kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequences += [[
            SNConv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.PReLU(), \
 \
            SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299 * input[:, 0, :, :] + .587 * input[:, 1, :, :] + .114 * input[:, 2, :, :]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2, -1))
        gradient = torch.cat([dx, dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3


# Defines the SegmentorHeadv2. Zero Padding and CSG for positional encoding.
class SegmentorHeadv2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, num_classes=19, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='zero'):
        super(SegmentorHeadv2, self).__init__()
        assert (n_blocks >= 0)
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(5, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        mult = 2 ** (n_downsampling)

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [ASPP(int(ngf), num_classes, norm_layer)]

        self.model = nn.Sequential(*model)
        self.csg = CatersianGrid()

    def forward(self, input):

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs, seg_fea = nn.parallel.data_parallel(self.model, torch.cat((input, self.csg(input)), 1), self.gpu_ids)
        else:
            outs, seg_fea = self.model(torch.cat((input, self.csg(input)), 1))
        return outs, seg_fea


class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id=None):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) for net in self.networks]

    def zero_grads(self, *args):
        if args is None:
            for opt in self.optimizers:
                opt.zero_grad()
        else:
            for dom in args:
                if dom < len(self.optimizers):
                    self.optimizers[dom].zero_grad()

    def step_grads(self, *args):
        if not len(args) == 0:
            for d in args:
                self.optimizers[d].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group in self.optimizers[dom_a].param_groups:
            param_group['lr'] = new_lr

        for param_group in self.optimizers[dom_b].param_groups:
            param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        if not isinstance(save_path, list):
            save_path = [save_path] * len(self.networks)
        for i, (net, path) in enumerate(zip(self.networks, save_path)):
            filename = path + f'{i}.pth'
            if isfile(filename):
                dic = torch.load(filename)
                # dic = {k: v for k, v in dic.items() if not '_sep' in k}
                net.load_state_dict(dic)


class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.enc_args = enc_args
        self.dec_args = dec_args
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append(self.shared_encoder)
            self.decoders.append(self.shared_decoder)
        self.networks = self.encoders + self.decoders

    # def init_optimizers(self, opt, lr, betas):
    #     self.optimizers = []
    #     for enc, dec in zip(self.encoders, self.decoders):
    #         params = itertools.chain(enc.parameters(), dec.parameters())
    #         self.optimizers.append(opt(params, lr=lr, betas=betas))

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) + '\n' + repr(d) + '\n' + \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) + '\n' + \
            'Number of parameters per Encoder: %d' % e_params + '\n' + \
            'Number of parameters per Deocder: %d' % d_params


class Color_G_Plexer(G_Plexer):
    def __init__(self, plexer: G_Plexer):
        super(Color_G_Plexer, self).__init__(0, 0, 0, 0, 0)
        self.enc_args = plexer.enc_args
        self.dec_args = plexer.dec_args
        self.encoders = plexer.encoders
        self.decoders = plexer.decoders
        self.sharing = plexer.sharing
        if self.sharing:
            self.shared_encoder = plexer.shared_encoder
            self.shared_decoder = plexer.shared_decoder
        self.networks = plexer.networks
        optimizers = plexer.optimizers[0]
        opt = optimizers.__class__
        lr = optimizers.param_groups[0]['lr']
        betas = optimizers.param_groups[0]['betas']
        concat_args = (256, 4, self.enc_args[3], self.enc_args[4], self.enc_args[6])
        self.feature_concatenation = ConcatBlock(*concat_args, gpu_ids=plexer.enc_args[5])
        # self.feature_concatenation = FusBlock(*concat_args, n_domains=2, gpu_ids=plexer.enc_args[5])
        # body = create_body(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), pretrained=True, n_in=4, cut=-2)
        # net_G = DynamicUnet(body, 3, (256, 256))
        # self.feature_concatenation = net_G
        self.networks.append(self.feature_concatenation)
        self.init_optimizers(opt, lr, betas)

    def to(self, device: torch.device):
        for net in self.networks:
            net.cuda(device)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) for net in self.encoders + self.decoders]
        self.optimizers.append(opt(self.feature_concatenation.parameters(), lr=lr, betas=betas))

    def train(self, *args, mode: bool = True):
        super(Color_G_Plexer, self).train(mode=mode)
        for net in self.networks:
            net.train(mode=False)
        for arg in args:
            if arg in [0, 1, 2, 3, 4, 5]:
                (self.encoders + self.decoders)[arg].train(mode=mode)
            elif arg == 6:
                self.feature_concatenation.train(mode=mode)
            else:
                raise ValueError('Invalid argument for training: %s' % arg)

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) + '\n' + repr(d) + '\n' + \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) + '\n' + \
            'Number of parameters per Encoder: %d' % e_params + '\n' + \
            'Number of parameters per Decoder: %d' % d_params

    # def fusion_features(self, feat1, feat2):  # The input need to be aligned
    #     return self.feature_concatenation(feat1, feat2)

    def fusion_features(self, image1, image2, mask=None):  # The input need to be aligned
        return self.feature_concatenation(image1, image2, mask)

    def separation_features(self, feat):
        return self.feature_concatenation(feat, sep=True)


class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def train(self, *args, mode: bool = True):
        super(D_Plexer, self).train(mode=mode)
        for net in self.networks:
            net.train(mode=False)
        for arg in args:
            if arg in [0, 1, 2]:
                self.networks[arg].train(mode=mode)
            else:
                pass

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) + '\n' + \
            'Created %d Discriminators' % len(self.networks) + '\n' + \
            'Number of parameters per Discriminator: %d' % t_params


class S_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(S_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=betas) \
                           for net in self.networks]

    def forward(self, input, domain):
        segmentor = self.networks[domain]
        return segmentor.forward(input)

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group_a in self.optimizers[dom_a].param_groups:
            param_group_a['lr'] = new_lr
            print('Learning rate of SegA is: %.4f.' % param_group_a['lr'])

        for param_group_b in self.optimizers[dom_b].param_groups:
            # print(param_group_b['lr'])
            param_group_b['lr'] = new_lr
            print('Learning rate of SegB is: %.4f.' % param_group_b['lr'])

    def train(self, *args, mode: bool = True):
        super(S_Plexer, self).train(mode=mode)
        for net in self.networks:
            net.train(mode=False)
        for arg in args:
            if arg in [0, 1, 2]:
                self.networks[arg].train(mode=mode)
            else:
                pass

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) + '\n' + \
            'Created %d Segmentors' % len(self.networks) + '\n' + \
            'Number of parameters per Segmentor: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:, domain, :, :] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x


class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;
                continue
            predictions.append(output[:, -1, :, :])
            if i != len(layers) - 1:
                input = output[:, :-1, :, :]
        return predictions
