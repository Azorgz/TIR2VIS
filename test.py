# import os
#
# import torch
# from einops import repeat
# from kornia.filters import median_blur
# from kornia.geometry import ScalePyramid, pyrup
# from numpy.random import permutation
# from torch import nn
# from torch.nn.functional import interpolate
# from torchmetrics.functional.image import image_gradients
# from torchvision.transforms.v2.functional import gaussian_blur
import math
import os

import torch
from torchvision.transforms.v2.functional import gaussian_blur

from ImagesCameras import ImageTensor as im
from models.utils_fct import detect_blob

# from ImagesCameras.Metrics import SSIM
# from ImagesCameras.tools.misc import time_fct


color_path = '/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/trainC/'
# path = '/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/trainC/FLIR_00137.png'
list_im_color = [color_path + p for p in sorted(os.listdir(color_path))]
for path in list_im_color:
    col = im(path)
    gray = col.GRAY()
    col_max, clr = col.max(dim=1, keepdim=True)
    # mask = detect_blob(col.max(dim=1)[0].squeeze().numpy(), min_radius=1, scale_blob=4)
    th = 0.08
    mask = (gray >= th*3) * (gray <= 1-th) + (col_max > gray * math.sqrt(2))
    mask = gaussian_blur(mask*1., [13, 13], [5, 5])
    filt_col = col * mask
    col.hstack(filt_col).show()


# ir_path = '/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/trainB/'
#
#
# list_im_ir = [ir_path + p for p in sorted(os.listdir(ir_path))]
#
# indexes = permutation(len(list_im_ir))
#
#
# class ScalePyramidFusion(ScalePyramid):
#
#     def __init__(self, n_levels: int = 3, init_sigma: float = 1.0, min_size: int = 15, double_image: bool = False) -> None:
#         super().__init__(n_levels=n_levels,
#                          init_sigma=init_sigma,
#                          min_size=min_size,
#                          double_image=double_image)
#         self.ssim_operator = SSIM(torch.device('cuda'))
#
#     def forward(self, L, ir):
#         L_pyr, sigmas,  scales = super().forward(L)
#         ir_pyr = super().forward(ir)[0]
#         ##
#         L_ref = [interpolate(L, scale_factor=0.5**i) for i in range(len(ir_pyr))]
#         L_DoG, L_base = self.compute_DoG_Base(L_pyr, L_ref)
#         ir_ref = [interpolate(ir, scale_factor=0.5**i) for i in range(len(ir_pyr))]
#         ir_DoG, ir_base = self.compute_DoG_Base(ir_pyr, ir_ref)
#         ##
#         base = [self.SSIM(ir_b, L_b) for ir_b, L_b in zip(ir_base, L_base)]
#         base = [b * ir_b + (1-b) * L_b for b, ir_b, L_b in zip(base, ir_base, L_base)]
#         #
#         grad_L_DoG = self.grad_DoG(L_DoG)
#         grad_ir_DoG = self.grad_DoG(ir_DoG)
#         details = [L_d[:, 0] * (grad_L > grad_ir) + ir_d[:, 0] * (grad_L <= grad_ir) for
#                    L_d, ir_d, grad_L, grad_ir in zip(L_DoG, ir_DoG, grad_L_DoG, grad_ir_DoG)]
#         res = self.fus_and_upscale(base, details)
#         return res
#
#     def fus_and_upscale(self, base: list, details: list):
#         b_ = None
#         for b, d in zip(reversed(base), reversed(details)):
#             if b_ is not None:
#                 b = torch.max(torch.cat([pyrup(b_), b], dim=1), dim=1, keepdim=True)[0]
#                 b_ = torch.sum(torch.cat([b, torch.sum(d, dim=1, keepdim=True)], dim=1), dim=1, keepdim=True)
#             else:
#                 b_ = torch.sum(torch.cat([b, torch.sum(d, dim=1, keepdim=True)], dim=1), dim=1, keepdim=True)
#         return b_
#
#     def grad_DoG(self, DoG):
#         res = []
#         for d in DoG:
#             dx, dy = image_gradients(d[:, 0])
#             magn = torch.sqrt(dx**2 + dy**2)
#             res.append((magn - magn.min())/(magn.max() - magn.min()))
#         return res
#
#     def SSIM(self, image_target, image_ref):
#         image_ssim = self.ssim_operator(image_target, image_ref, return_image=True)
#         return image_ssim.GRAY().resize(image_target.shape[-2:])
#
#     # def fus_upscale(self,images: list):
#     #     for image in reversed(images):
#
#     def compute_DoG_Base(self, pyr: list, ref:list):
#         DoG = []
#         base = []
#         pyr = [torch.cat([r.unsqueeze(2), p], dim=2) for (r, p) in zip(ref, pyr)]
#         for lv in pyr:
#             DoG.append(lv[:, :, :-1] - lv[:, :, 1:])
#             base.append(lv[:, :, -1])
#         return DoG, base
#
#
# def color_clamp(lab):
#     L, AB = lab[:, :1], lab[:, 1:]
#     mask = torch.where(L < L.mean() - L.std(), 0, 1)
#     mask_AB = repeat(mask, 'b () h w -> b c h w', c=2)
#     AB = AB * mask_AB + 0.5 * (1-mask_AB)
#     lab[:, 1:] = AB
#     return lab
#
#
# def intensity_fusion(color, ir, n_levels=7, init_sigma=0.8 ):
#     pyramid = ScalePyramidFusion(n_levels=n_levels, init_sigma=init_sigma)
#     L, AB = color[:, :1], color[:, 1:]
#     L_fused = pyramid(L, ir)
#     return im(L_fused)
#
#
# def SSIM_fus(image_target, image_ref):
#     ssim = SSIM(torch.device('cuda'))
#     image_ssim = ssim(image_target, image_ref, return_image=True)
#     return image_ssim.GRAY().resize(image_target.shape[-2:])
#
# for i in indexes:
#     color = im(list_im_color[i]).to('cuda')
#     ir = im(list_im_ir[i]).to('cuda')
#
#     lab = color.LAB()
#     ssim_mask = SSIM_fus(lab[:, :1], ir)**1.5
#     ssim_fused_L = lab[:, :1] * ssim_mask + (1-ssim_mask) * ir
#     lab = color_clamp(lab)
#     L = intensity_fusion(lab, ir)
#     lab[:, :1] = L
#     output1 = lab.RGB()
#     lab[:, :1] = ssim_fused_L
#     output2 = lab.RGB()
#     L = L.RGB('gray')
#     fused = im.hstack(L, ir.RGB('gray'), color, output1, output2)
#     fused.show()




