import functools
import math
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from kornia.filters import get_gaussian_kernel2d, filter2d, get_gaussian_kernel1d
from torch import nn, Tensor, conv2d
import skimage
from skimage import measure
from torch.nn import init, Conv2d, Softmax2d
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import cv2 as cv
import matplotlib.colors as mcolors
from torchvision.transforms.functional import gaussian_blur

from ImagesCameras import ImageTensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Spectral normalization base class
# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def UpdateSegGT(seg_tensor, ori_seg_GT, prob_th):
    "Use the high confidence predicted class to update original segmentation GT."

    sm = torch.nn.Softmax(dim=1)
    pred_sm = sm(seg_tensor.detach())
    pred_max_tensor = torch.max(pred_sm, dim=1)
    pred_max_value = pred_max_tensor[0]
    pred_max_category = pred_max_tensor[1]
    seg_HP_mask = torch.zeros_like(pred_max_value)
    seg_HP_mask = torch.where(pred_max_value > prob_th, torch.ones_like(pred_max_value), seg_HP_mask)
    seg_GT_float = ori_seg_GT.float()
    segGT_UC_mask = torch.zeros_like(seg_GT_float)
    segGT_UC_mask = torch.where(seg_GT_float == 255.0, torch.ones_like(seg_GT_float), segGT_UC_mask)
    seg_HP_mask_UC = seg_HP_mask.mul(segGT_UC_mask)
    mask_new_GT = seg_HP_mask_UC.mul(pred_max_category.float()) + (
            torch.ones_like(seg_HP_mask_UC) - seg_HP_mask_UC).mul(seg_GT_float)

    return mask_new_GT.detach()


def OnlSemDisModule(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th):
    """
    seg_tensor1:: realB - real IR segmentation pred
    seg_tensor2:: fakeA - fake RGB segmentation pred
    ori_seg_GT:: full 255 segmentation pred
    input_IR:: real_B_s -  real IR image
    prob_th: 0.75
    """
    "Online semantic distillation module: Use the common high confidence predicted class to update original segmentation GT."

    # Create the mask for the different predicted categories
    sm = torch.nn.Softmax(dim=1)
    pred_sm1 = sm(seg_tensor1.detach())
    pred_sm2 = sm(seg_tensor2.detach())
    pred_max_value1, pred_max_category1 = torch.max(pred_sm1, dim=1)
    pred_max_value2, pred_max_category2 = torch.max(pred_sm2, dim=1)
    mask_category1 = pred_max_category1.float()
    mask_category2 = pred_max_category2.float()

    # Find the intersection Area between the 2 masks
    mask_sub = mask_category1 - mask_category2
    mask_inter = (mask_sub == 0.0) * 1.
    seg_GT_float = ori_seg_GT.float()
    mask_inter_HP = (pred_max_value2 > prob_th) * (pred_max_value1 > prob_th)
    seg_inter_mask_UC = mask_inter * (seg_GT_float == 255.0)
    seg_inter_mask_UC_HP = mask_inter_HP.mul(seg_inter_mask_UC)

    mask_new_GT = (seg_inter_mask_UC_HP.mul(mask_category1) +
                   (torch.ones_like(seg_inter_mask_UC_HP) - seg_inter_mask_UC_HP).mul(seg_GT_float))
    mask_final = RefineIRMask(torch.squeeze(mask_new_GT), input_IR)
    ###Removal of vegetated areas from supervision
    mask_Bkg_all = (mask_final < 11.0) * 1.
    mask_Build_new = (mask_final == 2.0) * 1.
    mask_Sign_new = (mask_final == 6.0) * 1.
    mask_Light_new = (mask_final == 7.0) * 1.
    # mask_Car_new = (mask_final == 13.0) * 1.
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    "Before the parameters of the segmentation network are fixed, the threshold for the background category is set to 0.99; "
    "conversely, the threshold for all categories is set to 0.95."
    if torch.mean(mask_sub) == 0.0:
        High_th = prob_th
    else:
        High_th = prob_th + 0.04
    # High_th = 0.99
    LHP_mask = (pred_max_value1 < High_th) * 1.
    # VegRoad_LP_mask = LHP_mask.mul(mask_Veg_new) + LHP_mask.mul(mask_Road_new)
    VegRoad_LP_mask = LHP_mask.mul(mask_Bkg_stuff)
    ####Confusing categories Mask

    mask_CurtVeg = (torch.ones_like(mask_Bkg_stuff) - VegRoad_LP_mask).mul(mask_final) + VegRoad_LP_mask * 255.0

    return mask_CurtVeg.expand_as(ori_seg_GT).detach()


def UpdateIRSegGTv3(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th):
    """Combining the online semantic distillation module with masks (predicted offline) of object categories to update
    the segmentation pseudo-labels of NTIR images.
    ori_seg_GT: 1 * h * w."""

    sm = torch.nn.Softmax(dim=1)
    pred_sm1 = sm(seg_tensor1.detach())
    pred_sm2 = sm(seg_tensor2.detach())
    pred_max_value1, pred_max_category1 = torch.max(pred_sm1, dim=1)
    pred_max_value2, pred_max_category2 = torch.max(pred_sm2, dim=1)

    mask_category1 = pred_max_category1.float()
    mask_category2 = pred_max_category2.float()
    mask_sub = mask_category1 - mask_category2
    mask_inter = torch.where(mask_sub == 0.0, 1, 0)

    seg_HP_mask1 = torch.where(pred_max_value1 > prob_th, 1, 0)
    seg_HP_mask2 = torch.where(pred_max_value2 > prob_th, 1, 0)
    mask_inter_HP = seg_HP_mask1.mul(seg_HP_mask2)

    seg_inter_mask_UC_HP = mask_inter_HP.mul(mask_inter)

    mask_new_GT = seg_inter_mask_UC_HP.mul(mask_category1) + (1 - seg_inter_mask_UC_HP) * 255.0
    mask_final = RefineIRMask(torch.squeeze(mask_new_GT), input_IR)
    ###Removal of vegetated areas from supervision
    mask_Bkg_all = torch.where(mask_final < 11.0, 1., 0.)
    mask_Build_new = torch.where(mask_final == 2.0, 1., 0.)
    mask_Sign_new = torch.where(mask_final == 6.0, 1., 0.)
    mask_Light_new = torch.where(mask_final == 7.0, 1., 0.)
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    "Before the parameters of the segmentation network are fixed, the threshold for the background category is set to 0.99; "
    "conversely, the threshold for all categories is set to 0.95."
    if torch.mean(mask_sub) == 0.0:
        High_th = prob_th
    else:
        High_th = prob_th + 0.04

    # High_th = 0.99
    LHP_mask = torch.where(pred_max_value1 < High_th, 1., 0.)
    # VegRoad_LP_mask = LHP_mask.mul(mask_Veg_new) + LHP_mask.mul(mask_Road_new)
    VegRoad_LP_mask = LHP_mask.mul(mask_Bkg_stuff)
    ####Confusing categories Mask

    mask_CurtVeg = (1 - VegRoad_LP_mask).mul(mask_final) + VegRoad_LP_mask * 255.0

    ###Fusion with original GT masks of thing classes

    seg_GT_float = torch.squeeze(ori_seg_GT).float()
    segGT_obj_mask = torch.where(seg_GT_float < 255.0, 1., 0.)
    out_mask = (1 - segGT_obj_mask).mul(mask_CurtVeg) + segGT_obj_mask.mul(seg_GT_float)

    return out_mask.expand_as(ori_seg_GT).detach()


def RefineIRMask(ori_mask, input_IR):
    "Use original IR image to refine segmentaton mask for specific categories, i.e., Sky, Vegetation, Pole, and Person."
    "ori_mask: h * w,  input_IR: 1 * 3 * h * w"

    x_norm = (input_IR - torch.min(input_IR)) / (torch.max(input_IR) - torch.min(input_IR))
    IR_gray = torch.squeeze(.299 * x_norm[:, 0:1, :, :] + .587 * x_norm[:, 1:2, :, :] + .114 * x_norm[:, 2:3, :, :])

    Pole_mask = torch.where(ori_mask == 5.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Veg_mask = torch.where(ori_mask == 8.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Sky_mask = torch.where(ori_mask == 10.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Person_mask = torch.where(ori_mask == 11.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

    cnt_Pole = torch.sum(Pole_mask)
    cnt_Veg = torch.sum(Veg_mask)
    cnt_Sky = torch.sum(Sky_mask)
    cnt_Person = torch.sum(Person_mask)

    region_Pole = Pole_mask.mul(IR_gray)
    region_Veg = Veg_mask.mul(IR_gray)
    region_Sky = Sky_mask.mul(IR_gray)
    region_Person = Person_mask.mul(IR_gray)

    if cnt_Pole > 0:
        Pole_region_mean = torch.sum(region_Pole) / cnt_Pole
        if cnt_Sky > 0:
            Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
            #####Corrected Pole region mean.
            Pole_region_Corr_mean = (Pole_region_mean + Sky_region_mean) * 0.5
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_Corr_mean), 2))
        else:
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_mean), 2))

    if cnt_Veg > 0:
        Veg_region_mean = torch.sum(region_Veg) / cnt_Veg
        Veg_intradis = Veg_mask.mul(torch.pow((region_Veg - Veg_region_mean), 2))

    if cnt_Sky > 0:
        Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
        Sky_intradis = Sky_mask.mul(torch.pow((region_Sky - Sky_region_mean), 2))

    if cnt_Person > 0:
        Person_region_mean = torch.sum(region_Person) / cnt_Person
        Person_intradis = Person_mask.mul(torch.pow((region_Person - Person_region_mean), 2))

    ######Denoised for Sky
    if (cnt_Sky * cnt_Veg) > 0:
        Sky_Veg_dis = Sky_mask.mul(torch.pow((region_Sky - Veg_region_mean), 2))
        Sky_Veg_dis_err = Sky_intradis - Sky_Veg_dis
        Sky2Veg_mask = (Sky_Veg_dis_err > 0) * 1.
        mask_Sky_refine = Sky2Veg_mask * 255.0 + (Sky_mask - Sky2Veg_mask) * 10.0

        new_Sky_mask = Sky_mask - Sky2Veg_mask
        cnt_Sky_new = torch.sum(new_Sky_mask)
        region_Sky_new = new_Sky_mask.mul(IR_gray)
        if cnt_Sky_new > 0:
            Sky_region_mean_new = torch.sum(region_Sky_new) / cnt_Sky_new
        else:
            Sky_region_mean_new = Sky_region_mean
    elif cnt_Sky > 0:
        Sky_region_mean_new = Sky_region_mean
        mask_Sky_refine = Sky_mask * 10.0
    else:
        mask_Sky_refine = Sky_mask * 10.0
        # Sky_region_mean_new = Sky_region_mean

    ######Denoised for Pole
    if (cnt_Pole * cnt_Sky) > 0:
        Pole_Sky_dis = Pole_mask.mul(torch.pow((region_Pole - Sky_region_mean_new), 2))
        Pole_Sky_dis_err = Pole_intradis - Pole_Sky_dis
        Pole2Sky_mask = (Pole_Sky_dis_err > 0) * 1.
        mask_Pole_refine = Pole2Sky_mask * 255.0 + (Pole_mask - Pole2Sky_mask) * 5.0
    else:
        mask_Pole_refine = Pole_mask * 5.0

    ######Denoised for Person
    if (cnt_Person * cnt_Veg) > 0:
        Person_Veg_dis = Person_mask.mul(torch.pow((region_Person - Veg_region_mean), 2))
        Person_Veg_dis_err = Person_intradis - Person_Veg_dis
        Person2Veg_mask = (Person_Veg_dis_err > 0) * 1.
        mask_Person_refine = Person2Veg_mask * 255.0 + (Person_mask - Person2Veg_mask) * 11.0

        new_Person_mask = Person_mask - Person2Veg_mask
        cnt_Person_new = torch.sum(new_Person_mask)
        region_Person_new = new_Person_mask.mul(IR_gray)
        if cnt_Person_new > 0:
            Person_region_mean_new = torch.sum(region_Person_new) / cnt_Person_new
        else:
            Person_region_mean_new = Person_region_mean
    elif cnt_Person > 0:
        Person_region_mean_new = Person_region_mean
        mask_Person_refine = Person_mask * 11.0
    else:
        mask_Person_refine = Person_mask * 11.0
        # Person_region_mean_new = Person_region_mean

    ######Denoised for Vegetation
    if (cnt_Veg * cnt_Sky * cnt_Person) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        uncertain_mask_veg = torch.zeros_like(ori_mask)
        fuse_uncer = Veg2Sky_mask + Veg2Person_mask
        uncertain_mask_veg = torch.where(fuse_uncer > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        mask_Veg_refine = uncertain_mask_veg * 255.0 + (Veg_mask - uncertain_mask_veg) * 8.0
    elif (cnt_Veg * cnt_Sky) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Sky_mask * 255.0 + (Veg_mask - Veg2Sky_mask) * 8.0
    elif (cnt_Veg * cnt_Person) > 0:
        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Person_mask * 255.0 + (Veg_mask - Veg2Person_mask) * 8.0
    else:
        mask_Veg_refine = Veg_mask * 8.0

    mask_refine = mask_Sky_refine + mask_Pole_refine + mask_Person_refine + mask_Veg_refine + \
                  (torch.ones_like(ori_mask) - Pole_mask - Veg_mask - Sky_mask - Person_mask).mul(ori_mask)

    return mask_refine.detach()


def ClsMeanFea(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(b, 1, num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
                # temp_tensor = att_maps[0, i, :, :]
                if (torch.sum(temp_tensor)).item() > 0:
                    # print((torch.sum(temp_tensor)).item())
                    out_cls_tensor[i, 0] = 1.0
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)
                    # if torch.isnan(cls_fea_map).any().cpu().numpy():
                    #     print('NaN is existing in cls_fea_map. ')

                    ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(temp_tensor)  # b * c * 1 * 1
                    # ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w))
                    out_tensor[0, 0, i, :] = ave_fea

    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor_L2norm, out_cls_tensor


def ClsMeanPixelValue(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_ratio_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
                # temp_tensor = att_maps[0, i, :, :]
                if (torch.sum(temp_tensor)).item() > 0:
                    # print((torch.sum(temp_tensor)).item())
                    out_cls_tensor[i, 0] = 1.0
                    out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)

                    out_tensor[i, :] = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(
                        temp_tensor)  # b * c * 1 * 1
                    # ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w))
                    # out_tensor[i, :] = ave_fea

    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    # out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor, out_cls_tensor, out_cls_ratio_tensor


def ClsMeanPixelValuev2(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_ratio_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))

                if (torch.sum(temp_tensor)).item() > 0:
                    out_cls_tensor[i, 0] = 1.0
                    out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)

                    out_tensor[i, :] = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(
                        temp_tensor)  # b * c * 1 * 1

        temp_tensor_person = torch.zeros_like(seg_mask)
        temp_tensor_person = torch.where(seg_mask == 11, torch.ones_like(temp_tensor_person),
                                         torch.zeros_like(temp_tensor_person))
        temp_tensor_uncertain = torch.zeros_like(seg_mask)
        temp_tensor_uncertain = torch.where(seg_mask == 255, torch.ones_like(temp_tensor_person),
                                            torch.zeros_like(temp_tensor_person))
        temp_tensor_nonperson = torch.ones_like(temp_tensor_person) - temp_tensor_person - temp_tensor_uncertain

        if (torch.sum(temp_tensor_nonperson)).item() > 0:
            # out_cls_tensor[i, 0] = 1.0
            # out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
            nonperson_fea_map = (temp_tensor_nonperson.detach().expand_as(input_tensor)).mul(input_tensor)

            nonperson_fea_mean = (torch.squeeze(GAP(nonperson_fea_map) * h * w)) / torch.sum(temp_tensor_nonperson)
        else:
            nonperson_fea_mean = torch.zeros(1, c).cuda(gpu_ids)
    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    # out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor, out_cls_tensor, out_cls_ratio_tensor, nonperson_fea_mean


def getRoadDarkRegionMean(input_img, input_mask, gpu_ids=[]):
    "Obtain the mean value of the below-average brightness portion of the road area."
    img_gray = torch.squeeze(
        .299 * input_img[:, 0:1, :, :] + .587 * input_img[:, 1:2, :, :] + .114 * input_img[:, 2:3, :, :])
    Road_mask = (input_mask < 2.0) * 1.
    if torch.sum(Road_mask) > 0:
        Road_region = img_gray.mul(Road_mask.detach())
        Road_region_mean = torch.sum(Road_region) / torch.sum(Road_mask)
        Road_region_filling_one = Road_region + (torch.ones_like(input_mask) - Road_mask).mul(
            torch.ones_like(Road_region))
        Road_Dark_Region_Mask = torch.zeros_like(input_mask)
        Road_Dark_Region_Mask = torch.where(Road_region_filling_one < Road_region_mean, torch.ones_like(Road_mask),
                                            torch.zeros_like(Road_mask))
        out = torch.sum(Road_Dark_Region_Mask.mul(img_gray)) / torch.sum(Road_Dark_Region_Mask)
    else:
        out = torch.zeros(1).cuda(gpu_ids)

    return out


def getLightDarkRegionMean(cls_idx, input_img, input_mask, ref_img, gpu_ids=[]):
    "Obtain the mean value of the below-average brightness portion of the traffic light area."
    "The dark region mask of the traffic light region is first obtained using the reference image, and then "
    "the mean value of the corresponding region of the input image is calculated."

    _, _, h, w = input_img.size()
    input_img_gray = torch.squeeze(
        .299 * input_img[:, 0:1, :, :] + .587 * input_img[:, 1:2, :, :] + .114 * input_img[:, 2:3, :, :])
    ref_img_gray = torch.squeeze(
        .299 * ref_img[:, 0:1, :, :] + .587 * ref_img[:, 1:2, :, :] + .114 * ref_img[:, 2:3, :, :])
    light_mask_ori = torch.where(input_mask == cls_idx, 1., 0.).sum(dim=1, keepdim=True)
    max_pool_k3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    Light_mask = torch.squeeze(-max_pool_k3(- light_mask_ori))
    light_region_area = torch.sum(Light_mask)
    if light_region_area > 1:
        "In order to avoid that the bright and dark regions cannot be divided, the threshold is set to 1."
        Light_region = ref_img_gray.mul(Light_mask.detach())
        Light_region_mean = torch.sum(Light_region) / light_region_area

        Light_region_filling_one = Light_region + 1 - Light_mask
        Light_Dark_Region_Mask = torch.where(Light_region_filling_one < Light_region_mean, 1., 0.)
        Light_Dark_Region_Mean = torch.sum(Light_Dark_Region_Mask.mul(input_img_gray)) / torch.sum(
            Light_Dark_Region_Mask)

        Light_Bright_Region_Mask = Light_mask - Light_Dark_Region_Mask

        #####Light Bright region min
        Light_BR_filling_one = Light_Bright_Region_Mask.mul(input_img_gray) + 1 - Light_Bright_Region_Mask
        Light_Bright_Region_Min = torch.min(Light_BR_filling_one)
        ###Compute channle mean.
        input_img_3dim = torch.squeeze(input_img)
        input_img_DR_Masked = input_img_3dim.mul(Light_Dark_Region_Mask)
        input_img_DR_mean_3dim = torch.mean(input_img_DR_Masked, dim=0, keepdim=True)  #1*h*w
        input_img_DR_submean = (input_img_DR_Masked - input_img_DR_mean_3dim) ** 2
        input_img_DR_var = torch.max(torch.sum(input_img_DR_submean, dim=0))

    else:
        Light_Dark_Region_Mean = torch.zeros(1).cuda(gpu_ids)
        Light_Bright_Region_Min = torch.zeros(1).cuda(gpu_ids)
        input_img_DR_var = torch.zeros(1).cuda(gpu_ids)
        # Light_region_ref_var = torch.zeros(1).cuda(gpu_ids)

    return Light_Dark_Region_Mean, light_region_area, Light_Bright_Region_Min, input_img_DR_var


def getLightRegionColor(cls_idx, input_img, mask, gpu_ids=[]):
    "Obtain the color value of the traffic light."
    _, _, h, w = input_img.size()
    light_mask_ori = torch.where(mask == cls_idx, 1., 0.).sum(dim=1, keepdim=True)
    max_pool_k3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    Light_mask = torch.squeeze(-max_pool_k3(- light_mask_ori))
    light_region_area = torch.sum(Light_mask)
    coefflight_green = [-1.5, 0.5, 1]
    coefflight_orange = [-0.4, 0.7, -0.3]
    if light_region_area > 1:
        Light_region = input_img.mul(Light_mask.detach())
        Light_region_mean_channels = torch.mean(torch.mean(Light_region, dim=-1), dim=-1)
        sum_channel = (Light_region_mean_channels[:, 0] * coefflight_green[0] +
                       Light_region_mean_channels[:, 1] * coefflight_green[1] +
                       Light_region_mean_channels[:, 2] * coefflight_green[2])
        if sum_channel > 0:
            color = Tensor([0., 1., 0.2]).to(input_img.device)
        else:
            Light_region_mean_channels = torch.mean(torch.mean(Light_region, dim=-1), dim=-1)
            sum_channel = (Light_region_mean_channels[:, 0] * coefflight_orange[0] +
                           Light_region_mean_channels[:, 1] * coefflight_orange[1] +
                           Light_region_mean_channels[:, 2] * coefflight_orange[2])
            if sum_channel > 0:
                color = Tensor([1, 0.5, 0.]).to(input_img.device)
            else:
                color = Tensor([1, 0., 0.]).to(input_img.device)

    return color


def LightMaskDenoised(Seg_mask, real_vis, Avg_KernelSize, gpu_ids=[]):
    "Denoising of the traffic light mask region with given kernel size of average pooling."
    light_mask_ori = torch.where(Seg_mask == 6.0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))
    sky_mask = torch.where(Seg_mask == 10.0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))

    real_img_norm = (real_vis + 1.0) * 0.5
    real_vis_gray = torch.squeeze(
        .299 * real_img_norm[:, 0:1, :, :] + .587 * real_img_norm[:, 1:2, :, :] + .114 * real_img_norm[:, 2:3, :, :])
    real_vis_sky_region = sky_mask.mul(real_vis_gray)
    h, w = real_vis_gray.size()
    padsize = Avg_KernelSize // 2
    AvgPool_k5 = nn.AvgPool2d(Avg_KernelSize, stride=1, padding=padsize)
    real_vis_light_region = light_mask_ori.mul(real_vis_gray)
    real_vis_pooled = torch.squeeze(AvgPool_k5(real_vis_light_region.expand(1, 1, h, w)))

    "In a noisy traffic light region, if the distance between a given pixel and the average feature of the sky region is "
    "less than the distance between it and the average feature of the neighborhood in which it is located, the pixel has a "
    "high probability of belonging to the sky category, and therefore is set as a noisy pixel."
    if torch.sum(sky_mask) > 0:
        real_vis_sky_mean = torch.sum(real_vis_sky_region) / torch.sum(sky_mask)

        light_sky_dis = light_mask_ori.mul((real_vis_light_region - real_vis_sky_mean) ** 2)
        light_local_dis = light_mask_ori.mul((real_vis_light_region - real_vis_pooled) ** 2)
        sky_local_diff = light_local_dis - light_sky_dis
        sky_noise = torch.zeros_like(Seg_mask)
        sky_noise = torch.where(sky_local_diff > 0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))
        sky_noise_mask = light_mask_ori.mul(sky_noise)
    else:
        sky_noise_mask = torch.zeros_like(Seg_mask)

    light_mask_denoised = F.relu(light_mask_ori - sky_noise_mask)

    ####Filling the cavities inside the mask that are smaller than a certain area
    area_th = torch.sum(light_mask_ori) - torch.sum(light_mask_denoised)
    pre_mask = light_mask_denoised.cpu().numpy()
    pre_mask_rever = pre_mask <= 0
    pre_mask_rever = skimage.morphology.remove_small_objects(pre_mask_rever, min_size=area_th.item())
    pre_mask[pre_mask_rever <= 0] = 1
    light_mask_refine_tensor = torch.tensor(pre_mask).cuda(gpu_ids)
    out_mask = (torch.ones_like(Seg_mask) - light_mask_ori).mul(Seg_mask) + 6.0 * light_mask_refine_tensor + \
               255.0 * (light_mask_ori - light_mask_refine_tensor)

    return out_mask


def RefineLightMask(Seg_mask, real_vis, gpu_ids=[]):
    "Denoising of the traffic light mask region."
    Segmask_Light_DN_k5 = LightMaskDenoised(Seg_mask, real_vis, 5, gpu_ids)
    Segmask_Light_DN_k3 = LightMaskDenoised(Segmask_Light_DN_k5, real_vis, 3, gpu_ids)
    return torch.where(Segmask_Light_DN_k3 == 6.0, 1., 0.)


def PatchNormFea(input_array, sqrt_patch_num, gpu_ids=[]):
    "Calculate the L2 normalized features for each patch."
    h, w = input_array.size()

    crop_size = h // sqrt_patch_num
    pos_list = list(range(0, h, crop_size))
    patch_num = sqrt_patch_num * sqrt_patch_num
    patch_pixel = crop_size * crop_size
    out_fea_array = torch.zeros(patch_num, patch_pixel).cuda(gpu_ids)
    for p in range(sqrt_patch_num):
        for q in range(sqrt_patch_num):
            idx = p * sqrt_patch_num + q
            pos_h = pos_list[p]
            pos_w = pos_list[q]
            temp_patch = input_array[pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
            temp_patch_view = temp_patch.reshape(1, patch_pixel)
            out_fea_array[idx, :] = torch.div(temp_patch_view, (torch.norm(temp_patch_view) + 1e-4))

    return out_fea_array


def GetImgHieFea(input_rgb, input_gray, value_th_list, th_num, gpu_ids=[]):
    "Get the hierarchical features of RGB images with different value intervals."
    _, c, h, w = input_rgb.size()
    # th_mask_tensor = torch.zeros(th_num, h, w).cuda(gpu_ids)
    out_rgb_fea_tensor = torch.zeros(th_num, c).cuda(gpu_ids)
    out_dist_pixels_tensor = torch.zeros(th_num, 1).cuda(gpu_ids)
    # mask_squeeze = torch.squeeze(input_mask)
    gray_img_squeeze = torch.squeeze(input_gray)
    GAP = nn.AdaptiveAvgPool2d(1)
    for i in range(th_num):
        temp_mask1 = torch.where(gray_img_squeeze < value_th_list[i], 1., 0.)
        temp_mask2 = torch.where(gray_img_squeeze < value_th_list[i + 1], 1., 0.)
        th_mask = temp_mask2 - temp_mask1
        mask_pixels = torch.sum(th_mask)
        out_dist_pixels_tensor[i, :] = mask_pixels
        if mask_pixels != 0:
            rgb_mask = input_rgb.mul(th_mask.expand_as(input_rgb))
            out_rgb_fea_tensor[i, :] = (torch.squeeze(GAP(rgb_mask) * h * w)) / mask_pixels

    return out_rgb_fea_tensor, out_dist_pixels_tensor


def bhw_to_onehot(bhw_tensor1, num_classes, gpu_ids):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes: 20 (19 + uncertain_clsidx)
    Returns: b,num_classes,h,w
    """
    assert bhw_tensor1.ndim == 3, bhw_tensor1.shape
    # assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    # bhw_tensor = bhw_tensor1
    # bhw_tensor[(bhw_tensor == 255)] = 5

    bhw_tensor = torch.zeros_like(bhw_tensor1).cuda(gpu_ids)
    uncertain_clsidx = num_classes - 1
    bhw_tensor = torch.where(bhw_tensor1 == 255, uncertain_clsidx, bhw_tensor1)
    # one_hot = torch.eye(num_classes).index_select(dim=0, index=bhw_tensor.reshape(-1)).cuda(gpu_ids)
    one_hot = torch.eye(num_classes).cuda(gpu_ids).index_select(dim=0, index=bhw_tensor.reshape(-1))
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    out_tensor = one_hot.permute(0, 3, 1, 2)

    return out_tensor[:, :-1, :, :]


def GetFeaMatrixCenter(fea_array, cluster_num, max_iter, gpu_ids):
    "Obtain the central features of each cluster of the feature matrix."
    # if gpu_ids == 0:
    # kmeans
    _, cluster_centers = kmeans(
        X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device(f'cuda:{gpu_ids}'),
        tqdm_flag=False, iter_limit=max_iter)

    return cluster_centers.cuda(gpu_ids)


def FakeIRFGMergeMask(vis_segmask, IR_seg_tensor, gpu_ids):
    "Selecting a suitable foreground mask from the fake IR image and fuse it with the real IR image."

    sm = torch.nn.Softmax(dim=1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    vis_FG_idx_list = [6, 7, 11, 13, 14, 15, 16, 17]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    IR_road_mask = np.zeros_like(real_IR_segmask)
    IR_road_mask = np.where(real_IR_segmask < 2.0, 1.0, 0.0)
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(vis_GT_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(temp_connect_mask) > 50:
                if vis_FG_idx_list[i] in traffic_sign_list:
                    output_FG_Mask += temp_connect_mask
                elif vis_FG_idx_list[i] in large_FG_list:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
                else:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
    # print(np.sum(output_FG_Mask))

    return torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)


def FakeIRFGMergeMaskv3(vis_segmask, IR_seg_tensor, real_vis, fake_IR, gpu_ids):
    "Selecting a suitable foreground mask from the fake IR image and fusing it with the real IR image, "
    "and keeping the original foreground area unchanged. Vertical flipping of traffic light areas that "
    "meet the conditions to reduce the wrong color of traffic lights due to uneven distribution of red "
    "and green lights."

    sm = torch.nn.Softmax(dim=1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    _, _, h, w = IR_seg_tensor.size()
    vis_FG_idx_list = [6, 7, 17]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    real_vis_numpy = torch.squeeze(real_vis).detach().cpu().numpy()
    fake_IR_numpy = torch.squeeze(fake_IR).detach().cpu().numpy()
    IR_road_mask = np.where(real_IR_segmask < 2.0, 1.0, 0.0)
    IR_FG1_mask = np.where(real_IR_segmask > 10.0, 1.0, 0.0)
    IR_light_mask = np.where(real_IR_segmask == 6.0, 1.0, 0.0)
    IR_sign_mask = np.where(real_IR_segmask == 7.0, 1.0, 0.0)
    IR_FG_mask = IR_FG1_mask + IR_light_mask + IR_sign_mask
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    output_FG_Mask_ori = np.zeros_like(real_IR_segmask)
    output_HL_Mask = np.zeros_like(real_IR_segmask)
    output_Light_TopMask = np.zeros_like(real_IR_segmask)
    output_Light_BottomMask = np.zeros_like(real_IR_segmask)
    output_FG_FakeIR = np.zeros_like(fake_IR_numpy)
    output_FG_RealVis = np.zeros_like(real_vis_numpy)
    for i, idx in enumerate(vis_FG_idx_list):
        #######erode
        temp_mask_ori = np.where(vis_GT_segmask == idx, 1.0, 0.0)
        max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        temp_mask_erode = -max_pool_k3(-torch.Tensor(temp_mask_ori).unsqueeze(0).unsqueeze(0))
        temp_mask = torch.squeeze(temp_mask_erode).numpy()
        #########
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask
            FG_mask_overlap = temp_connect_mask * IR_FG_mask
            fake_IR_masked = temp_connect_mask * fake_IR_numpy
            real_vis_masked = temp_connect_mask * real_vis_numpy
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if idx in traffic_sign_list:
                        if i == 0:
                            temp_FG_Mask, temp_FG_FakeIR, temp_FG_RealVis, temp_highlight_mask, temp_TopMask, temp_BottomMask = ObtainTLightMixedMask(
                                temp_connect_mask, fake_IR_masked, real_vis_masked, h)
                            output_FG_Mask += temp_FG_Mask
                            output_FG_FakeIR += temp_FG_FakeIR
                            output_FG_RealVis += temp_FG_RealVis
                            output_HL_Mask += temp_highlight_mask
                            output_Light_TopMask += temp_TopMask
                            output_Light_BottomMask += temp_BottomMask
                        else:
                            output_FG_Mask += temp_connect_mask
                            output_FG_FakeIR += fake_IR_masked
                            output_FG_RealVis += real_vis_masked
                            output_FG_Mask_ori += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask
                            output_FG_FakeIR += fake_IR_masked
                            output_FG_RealVis += real_vis_masked
                            output_FG_Mask_ori += temp_connect_mask
    # print(np.sum(output_FG_Mask))

    out_FG_mask = torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_FakeIR = torch.tensor(output_FG_FakeIR).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_RealVis = torch.tensor(output_FG_RealVis).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_mask_ori = torch.tensor(output_FG_Mask_ori).cuda(gpu_ids).expand(1, 3, 256, 256)

    #########Flip fake mask
    vis_GT_flip = torch.flip(torch.squeeze(vis_segmask), dims=[1])
    vis_GT_flip_segmask = vis_GT_flip.float().detach().cpu().numpy()
    output_FG_Mask_Flip = np.zeros_like(real_IR_segmask)
    output_FG_FakeIR_Flip = np.zeros_like(fake_IR_numpy)
    output_FG_RealVis_Flip = np.zeros_like(real_vis_numpy)
    # output_HL_Mask_Flip = np.zeros_like(real_IR_segmask)
    fake_IR_numpy_Flip = torch.squeeze(torch.flip(fake_IR, dims=[3])).detach().cpu().numpy()
    real_vis_numpy_Flip = torch.squeeze(torch.flip(real_vis, dims=[3])).detach().cpu().numpy()
    IR_FG_mask_update = IR_FG_mask + output_FG_Mask
    IR_road_mask_update = IR_road_mask - IR_road_mask * output_FG_Mask
    for i in range(len(vis_FG_idx_list)):

        #######erode
        temp_mask_ori = np.zeros_like(vis_GT_flip_segmask)
        temp_mask_ori = np.where(vis_GT_flip_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        temp_mask_erode = -max_pool_k3(-torch.Tensor(temp_mask_ori).unsqueeze(0).unsqueeze(0))
        temp_mask = torch.squeeze(temp_mask_erode).numpy()
        #########
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask_update
            FG_mask_overlap = temp_connect_mask * IR_FG_mask_update

            fake_IR_masked_Flip = temp_connect_mask * fake_IR_numpy_Flip
            real_vis_masked_Flip = temp_connect_mask * real_vis_numpy_Flip
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        if i == 0:
                            temp_FG_Mask2, temp_FG_FakeIR2, temp_FG_RealVis2, temp_highlight_mask2, temp_TopMask2, temp_BottomMask2 = ObtainTLightMixedMask(
                                temp_connect_mask, fake_IR_masked_Flip, real_vis_masked_Flip, h)
                            output_FG_Mask_Flip += temp_FG_Mask2
                            output_FG_FakeIR_Flip += temp_FG_FakeIR2
                            output_FG_RealVis_Flip += temp_FG_RealVis2
                            output_HL_Mask += temp_highlight_mask2
                            output_Light_TopMask += temp_TopMask2
                            output_Light_BottomMask += temp_BottomMask2
                        else:
                            output_FG_Mask_Flip += temp_connect_mask
                            output_FG_FakeIR_Flip += fake_IR_masked_Flip
                            output_FG_RealVis_Flip += real_vis_masked_Flip
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask
                            output_FG_FakeIR_Flip += fake_IR_masked_Flip
                            output_FG_RealVis_Flip += real_vis_masked_Flip

    out_FG_mask_flip = torch.tensor(output_FG_Mask_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_FakeIR_flip = torch.tensor(output_FG_FakeIR_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_RealVis_flip = torch.tensor(output_FG_RealVis_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_HL_mask = torch.tensor(output_HL_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_Light_TopMask = torch.tensor(output_Light_TopMask).cuda(gpu_ids).expand(1, 256, 256)
    out_Light_BottomMask = torch.tensor(output_Light_BottomMask).cuda(gpu_ids).expand(1, 256, 256)
    out_Light_mask = torch.cat([out_Light_TopMask, out_Light_BottomMask], dim=0)

    return out_FG_mask, out_FG_FakeIR, out_FG_RealVis, out_FG_mask_flip, out_FG_FakeIR_flip, \
        out_FG_RealVis_flip, out_FG_mask_ori, out_HL_mask, out_Light_mask


def ObtainTLightMixedMask(temp_connect_mask, fake_IR_masked, real_vis_masked, patch_height):
    "Extraction of traffic light related areas and masks."

    temp_connect_mask_row_sum = np.sum(temp_connect_mask, axis=1)
    temp_connect_mask_col_sum = np.sum(temp_connect_mask, axis=0)
    region_AspectRatio = np.max(temp_connect_mask_col_sum) / np.max(temp_connect_mask_row_sum)
    _, temp_r2g_area_ori, temp_red_mask_ori = Red2Green(real_vis_masked, temp_connect_mask)
    _, temp_g2r_area_ori, temp_green_mask_ori = Green2Red(real_vis_masked, temp_connect_mask)

    row_pos_array = np.matmul(np.arange(patch_height).reshape((patch_height, 1)), np.ones((1, patch_height)))
    mask_pos = temp_connect_mask * row_pos_array
    mask_pos_padding_h = temp_connect_mask * row_pos_array + \
                         (np.ones_like(temp_connect_mask) - temp_connect_mask) * patch_height
    mask_pos_row_min = int(mask_pos_padding_h.min())
    mask_pos_row_max = int(mask_pos.max())
    mask_mid_row = (mask_pos_row_min + mask_pos_row_max) // 2
    top_mask = np.zeros_like(temp_connect_mask)
    top_mask[mask_pos_row_min:(mask_mid_row + 1), :] = 1.0
    bottom_mask = np.ones_like(temp_connect_mask) - top_mask
    # IoU_th = 0.8
    IoU_th = 0.5
    if region_AspectRatio > 1.75:
        "When the aspect ratio of a given traffic light instance is greater than a given threshold, vertical flip and "
        "transition between red and green lights are performed."
        ver_flip_idx = torch.rand(1)
        temp_VerFlip_mask, temp_VerFlip_fakeIR, temp_VerFlip_realVis = LocalVerticalFlip(temp_connect_mask,
                                                                                         fake_IR_masked,
                                                                                         real_vis_masked,
                                                                                         mask_pos_row_min,
                                                                                         mask_pos_row_max)
        temp_r2g_vis, temp_r2g_area, temp_red_mask = Red2Green(temp_VerFlip_realVis, temp_VerFlip_mask)
        temp_g2r_vis, temp_g2r_area, temp_green_mask = Green2Red(temp_VerFlip_realVis, temp_VerFlip_mask)
        DLS_idx = np.random.random(1)
        Decay_factor = 1.0
        if ver_flip_idx > 0.5:

            #########Generating double light spot
            # DLS_idx = torch.rand(1)
            # Decay_factor = torch.rand(1)
            if temp_r2g_area > temp_g2r_area:
                vis_GT_masked = (temp_r2g_vis - 0.5) * 2.0
                output_FG_RealVis = vis_GT_masked
                output_highlight_mask = temp_red_mask

                if DLS_idx > 0.5:
                    "In the traffic light instance of NTIR image, a double light spot situation (i.e., the temperature of "
                    "both the illuminated main signal and the unilluminated auxiliary signal is higher) is usually presented "
                    "due to the high frequency of illumination of the red and green lights."

                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        "When the traffic light instance mask is approximately rectangular, i.e., the IoU of the vertically "
                        "flipped mask with respect to the original mask is greater than a given threshold, a two-spot traffic "
                        "light is synthesized in the pseudo-NTIR image."

                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        # print('Case1.')
                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (
                                top_mask * temp_connect_mask * fake_IR_masked)
                        bottom_fake_IR_masked = bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        # print('Case2.')
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    # print('Case3.')
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR

            else:
                vis_GT_masked = (temp_g2r_vis - 0.5) * 2.0
                output_FG_RealVis = vis_GT_masked
                output_highlight_mask = temp_green_mask

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        # print('Case4.')
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (
                                bottom_mask * temp_connect_mask * fake_IR_masked)
                        top_fake_IR_masked = top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR

                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = bottom_mask * temp_connect_mask + top_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        # print('Case5.')
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    # print('Case6.')
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR
            ####################
        else:
            output_FG_RealVis = real_vis_masked
            # print('Case2. \n')
            if temp_r2g_area_ori > temp_g2r_area_ori:
                output_highlight_mask = temp_red_mask_ori

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.

                        top_fake_IR_masked = top_mask * temp_connect_mask * fake_IR_masked
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (
                                bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked
            else:
                output_highlight_mask = temp_green_mask_ori

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.

                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (
                                top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        bottom_fake_IR_masked = bottom_mask * temp_connect_mask * fake_IR_masked
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_VerFlip_mask + bottom_mask * temp_connect_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked
    else:
        output_FG_Mask = temp_connect_mask
        output_FG_FakeIR = fake_IR_masked
        output_FG_RealVis = real_vis_masked
        if temp_r2g_area_ori > temp_g2r_area_ori:
            output_highlight_mask = temp_red_mask_ori
        else:
            output_highlight_mask = temp_green_mask_ori

    output_FG_top_mask = output_FG_Mask * top_mask
    output_FG_bottom_mask = output_FG_Mask * bottom_mask
    # print(output_FG_FakeIR.shape)
    return output_FG_Mask, output_FG_FakeIR, output_FG_RealVis, output_highlight_mask, output_FG_top_mask, output_FG_bottom_mask


def ComIoUNumpy(input_mask1, input_mask2):
    "input_mask:h*w. Numpy array."
    mask_inter = input_mask1 * input_mask2
    mask_fused = input_mask1 + input_mask2
    mask_union = np.zeros_like(input_mask1)
    mask_union = np.where(mask_fused > 0.0, 1.0, 0.0)
    res_IoU = np.sum(mask_inter) / np.sum(mask_union)

    return res_IoU


def LocalVerticalFlip(input_mask, fake_IR, real_vis, region_min_row, region_max_row):
    "input_mask:h*w. fake_IR:c*h*w. Numpy array."

    h = input_mask.shape[0]
    # w = input_mask.shape[1]
    input_mask_flip = np.flip(input_mask, 0)
    fake_IR_flip = np.flip(fake_IR, 1)
    real_vis_flip = np.flip(real_vis, 1)
    flip_region_min_row = int(h - region_max_row - 1)
    flip_region_max_row = int(h - region_min_row - 1)
    output_mask = np.zeros_like(input_mask)
    output_fake_IR = np.zeros_like(fake_IR)
    output_real_vis = np.zeros_like(real_vis)

    output_mask[region_min_row:region_max_row + 1, :] = input_mask_flip[flip_region_min_row:flip_region_max_row + 1, :]
    output_fake_IR[:, region_min_row:region_max_row + 1, :] = fake_IR_flip[:,
                                                              flip_region_min_row:flip_region_max_row + 1, :]
    output_real_vis[:, region_min_row:region_max_row + 1, :] = real_vis_flip[:,
                                                               flip_region_min_row:flip_region_max_row + 1, :]

    return output_mask, output_fake_IR, output_real_vis


def Red2Green(input_rgb, input_mask):
    "input_rgb: c*h*w. input_mask: h*w. Numpy array."
    input_rgb_norm = (input_rgb + 1.0) * 0.5
    input_rgb_masked = input_rgb_norm * input_mask
    # input_hsv = rgb_to_hsv(torch.tensor(input_rgb_masked).unsqueeze(0))
    input_hsv = rgb_to_hsv(torch.Tensor(input_rgb_masked).unsqueeze(0))
    input_hsv_numpy = torch.squeeze(input_hsv).numpy()
    # print(input_hsv_numpy.shape)
    input_h = input_hsv_numpy[0, :, :] * 180.0
    input_s = input_hsv_numpy[1, :, :] * 255.0
    input_v = input_hsv_numpy[2, :, :] * 255.0
    # s_mask = np.zeros_like(input_s)
    s_mask = np.where(input_s > 42, 1.0, 0.0)
    v_mask = np.where(input_v > 45, 1.0, 0.0)
    h_mask1 = np.where(input_h < 25, 1.0, 0.0)
    h_mask2 = np.where(input_h > 155, 1.0, 0.0)
    #########
    red_mask1_ori = s_mask * v_mask * h_mask1
    red_mask2_ori = s_mask * v_mask * h_mask2
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # HL_Mask_dilate = max_pool_k5(HL_Mask)
    red_mask1_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(red_mask1_ori).unsqueeze(0).unsqueeze(0)))
    red_mask2_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(red_mask2_ori).unsqueeze(0).unsqueeze(0)))
    red_mask1 = torch.squeeze(red_mask1_dilate).numpy()
    red_mask2 = torch.squeeze(red_mask2_dilate).numpy()
    #############
    red_mask_area1 = np.sum(red_mask1)
    red_mask_area2 = np.sum(red_mask2)
    red_mask_area = red_mask_area1 + red_mask_area2
    red_mask_fused = red_mask1 + red_mask2
    if red_mask_area1 > 0:
        h_r2g_out1 = red_mask1 * ((input_h * 0.12) + 87.0)
    else:
        h_r2g_out1 = np.zeros_like(input_h)

    if red_mask_area2 > 0:
        h_r2g_out2 = red_mask2 * ((input_h * 0.12) + 64.0)
    else:
        h_r2g_out2 = np.zeros_like(input_h)

    h_out = (np.ones_like(input_h) - red_mask1 - red_mask2) * input_h + h_r2g_out1 + h_r2g_out2
    hsv_out = np.zeros_like(input_hsv_numpy)
    hsv_out[0, :, :] = h_out / 180.0
    s_out = (np.ones_like(input_h) - red_mask1 - red_mask2) * input_hsv_numpy[1, :, :] + (red_mask1 + \
                                                                                          red_mask2) * (
                    input_hsv_numpy[1, :, :] * 0.5)
    hsv_out[1, :, :] = s_out
    hsv_out[2, :, :] = input_hsv_numpy[2, :, :]
    # out_r2g = hsv_to_rgb(torch.tensor(hsv_out).unsqueeze(0))
    out_r2g = hsv_to_rgb(torch.Tensor(hsv_out).unsqueeze(0))
    res_numpy = torch.squeeze(out_r2g).numpy()

    return res_numpy, red_mask_area, red_mask_fused


def Green2Red(input_rgb, input_mask):
    "input_rgb: c*h*w. input_mask: h*w. Numpy array."
    input_rgb_norm = (input_rgb + 1.0) * 0.5
    input_rgb_masked = input_rgb_norm * input_mask
    # input_hsv = rgb_to_hsv(torch.tensor(input_rgb_masked).unsqueeze(0))
    input_hsv = rgb_to_hsv(torch.Tensor(input_rgb_masked).unsqueeze(0))
    input_hsv_numpy = torch.squeeze(input_hsv).numpy()
    input_h = input_hsv_numpy[0, :, :] * 180.0
    input_s = input_hsv_numpy[1, :, :] * 255.0
    input_v = input_hsv_numpy[2, :, :] * 255.0
    # s_mask = np.zeros_like(input_s)
    s_mask = np.where(input_s > 25, 1.0, 0.0)
    v_mask = np.where(input_v > 45, 1.0, 0.0)
    h_mask1 = np.where(input_h < 90, 1.0, 0.0)
    h_mask2 = np.where(input_h > 67, 1.0, 0.0)
    h_mask3 = np.where(input_h > 90, 1.0, 0.0)
    h_mask4 = np.where(input_h < 110, 1.0, 0.0)
    ######Padding
    green_mask1_ori = s_mask * v_mask * h_mask1 * h_mask2
    green_mask2_ori = s_mask * v_mask * h_mask3 * h_mask4
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # HL_Mask_dilate = max_pool_k5(HL_Mask)
    green_mask1_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(green_mask1_ori).unsqueeze(0).unsqueeze(0)))
    green_mask2_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(green_mask2_ori).unsqueeze(0).unsqueeze(0)))
    green_mask1 = torch.squeeze(green_mask1_dilate).numpy()
    green_mask2 = torch.squeeze(green_mask2_dilate).numpy()
    ##############
    green_mask_area1 = np.sum(green_mask1)
    green_mask_area2 = np.sum(green_mask2)
    green_mask_area = green_mask_area1 + green_mask_area2
    green_mask_fused = green_mask1 + green_mask2
    if green_mask_area1 > 0:
        h_g2r_out1 = green_mask1 * ((input_h * 0.5) - 33.5)
    else:
        h_g2r_out1 = np.zeros_like(input_h)

    if green_mask_area2 > 0:
        h_g2r_out2 = green_mask2 * ((input_h * (-0.5)) + 55.0)
    else:
        h_g2r_out2 = np.zeros_like(input_h)

    h_out = (np.ones_like(input_h) - green_mask1 - green_mask2) * input_h + h_g2r_out1 + h_g2r_out2
    hsv_out = np.zeros_like(input_hsv_numpy)
    hsv_out[0, :, :] = h_out / 180.0
    s_out = (np.ones_like(input_h) - green_mask1 - green_mask2) * input_hsv_numpy[1, :, :] + (green_mask1 + \
                                                                                              green_mask2) * (
                    input_hsv_numpy[1, :, :] * 4.0)
    hsv_out[1, :, :] = s_out
    hsv_out[2, :, :] = input_hsv_numpy[2, :, :]
    # out_g2r = hsv_to_rgb(torch.tensor(hsv_out).unsqueeze(0))
    out_g2r = hsv_to_rgb(torch.Tensor(hsv_out).unsqueeze(0))
    res_numpy = torch.squeeze(out_g2r).numpy()

    return res_numpy, green_mask_area, green_mask_fused


def FakeVisFGMergeMask(IR_seg_tensor, vis_segmask, gpu_ids):
    "Selecting a suitable foreground mask from the fake visible image and fusing it with the real visible image, "
    "and keeping the original foreground area unchanged."

    sm = torch.nn.Softmax(dim=1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    # vis_FG_idx_list = [6, 7, 11, 12, 14, 15, 16, 17]
    vis_FG_idx_list = [6, 7, 17]
    # vis_FG_idx_list = [6, 7]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    vis_road_mask = np.where(vis_GT_segmask < 2.0, 1.0, 0.0)
    vis_FG1_mask = np.where(vis_GT_segmask > 10.0, 1.0, 0.0)
    vis_light_mask = np.where(vis_GT_segmask == 6.0, 1.0, 0.0)
    vis_sign_mask = np.where(vis_GT_segmask == 7.0, 1.0, 0.0)
    vis_FG_mask = vis_FG1_mask + vis_light_mask + vis_sign_mask
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.where(real_IR_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * vis_road_mask
            FG_mask_overlap = temp_connect_mask * vis_FG_mask
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        output_FG_Mask += temp_connect_mask
                    elif vis_FG_idx_list[i] in large_FG_list:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask

    out_FG_mask = torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)

    #########Flip fake mask
    IR_mask_flip = torch.flip(torch.squeeze(IR_segmask), dims=[1])
    IR_flip_segmask = IR_mask_flip.float().detach().cpu().numpy()
    output_FG_Mask_Flip = np.zeros_like(real_IR_segmask)
    vis_FG_mask_update = vis_FG_mask + output_FG_Mask
    vis_road_mask_update = vis_road_mask - vis_road_mask * output_FG_Mask
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(IR_flip_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num + 1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * vis_road_mask_update
            FG_mask_overlap = temp_connect_mask * vis_FG_mask_update
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        output_FG_Mask_Flip += temp_connect_mask
                    elif vis_FG_idx_list[i] in large_FG_list:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask

    out_FG_mask_flip = torch.tensor(output_FG_Mask_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)

    return out_FG_mask, out_FG_mask_flip, IR_segmask


def UpdateFakeIRSegGT(fake_IR, Seg_mask, dis_th):
    "The GT corresponding to the high-brightness region in the vegetation area in the fake IR image is set as "
    "an uncertain region, which is to reduce the perception of the street light as vegetation in the "
    "real IR image."

    _, _, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest').squeeze()
    veg_mask = torch.where(GT_mask == 8.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    fake_IR_gray = torch.squeeze(
        .299 * fake_img_norm[:, 0:1, :, :] + .587 * fake_img_norm[:, 1:2, :, :] + .114 * fake_img_norm[:, 2:3, :, :])

    if torch.sum(veg_mask) > 0:
        region_veg = veg_mask.mul(fake_IR_gray)
        region_veg_mean = torch.sum(region_veg) / torch.sum(veg_mask)
        region_veg_max = torch.max(region_veg)
        veg_range_high_ratio = (region_veg_max - region_veg_mean) / (region_veg_mean + 1e-4)

        "If the difference between the maximum brightness value and the average brightness value of a vegetation region is "
        "greater than a given threshold, the semantic labeling of the corresponding bright region (i.e., the region with "
        "greater than average brightness) is set to uncertain."
        if veg_range_high_ratio > dis_th:
            veg_high_mask = torch.where(region_veg > region_veg_mean, torch.ones_like(GT_mask),
                                        torch.zeros_like(GT_mask))
            mask_new_GT = veg_high_mask * 255.0 + (torch.ones_like(veg_high_mask) - veg_high_mask).mul(GT_mask)
            out_mask = mask_new_GT.expand(1, h, w)
        else:
            out_mask = Seg_mask
    else:
        out_mask = Seg_mask

    return out_mask


def UpdateFakeVISSegGT(real_vis_night, Seg_mask, dis_lum):
    "The GT corresponding to the low luminosity regions in the sky area in the fake RGB image is set as"
    "an uncertain region, which is to reduce the perception of the street light as vegetation in the "
    "real RGB-Night image."

    _, _, h, w = real_vis_night.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest').squeeze()
    veg_mask = torch.where(GT_mask == 8.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    real_vis_night_norm = (real_vis_night + 1.0) * 0.5
    real_vis_night_gray = torch.squeeze(
        .299 * real_vis_night_norm[:, 0:1, :, :] + .587 * real_vis_night_norm[:, 1:2, :,
                                                          :] + .114 * real_vis_night_norm[:, 2:3, :, :])
    veg_gray = veg_mask * real_vis_night_gray
    mask_high_light = torch.where(veg_gray > dis_lum, torch.ones_like(Seg_mask) * 255., Seg_mask)
    mask_low_light = torch.where(veg_gray < dis_lum, torch.ones_like(Seg_mask) * 255., mask_high_light)
    return mask_low_light


def IRComPreProcessv6(FG_mask, FG_mask_flip, Fake_IR_masked_ori, Fake_IR_masked_flip_ori, Real_IR, Real_IR_SegMask,
                      HL_Mask_ori):
    "Gaussian blurring is applied to the fake NTIR images to enhance the plausibility of the appearance of FG region."

    #### HL_Mask padding
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    HL_Mask = -max_pool_k5(-max_pool_k5(HL_Mask_ori))

    if torch.sum(HL_Mask) > 0:
        FG_mask_sub_HL = FG_mask - HL_Mask
        FG_LL_mask = torch.where(FG_mask_sub_HL > 0.0, 1., 0.)

        FG_mask_sub_HL_flip = FG_mask_flip - HL_Mask
        FG_LL_mask_flip = torch.where(FG_mask_sub_HL_flip > 0.0, 1., 0.)
    else:
        FG_LL_mask = FG_mask
        FG_LL_mask_flip = FG_mask_flip

    ######Calculate the mean value of the road area
    IR_SegMask = torch.squeeze(Real_IR_SegMask.float())
    RealIR_RoadMask = torch.where(IR_SegMask == 0.0, torch.ones_like(IR_SegMask), torch.zeros_like(IR_SegMask))
    Real_IR_norm = (Real_IR + 1.0) * 0.5
    Fake_IR_norm = (Fake_IR_masked_ori + 1.0) * 0.5
    Fake_IR_flip_norm = (Fake_IR_masked_flip_ori + 1.0) * 0.5
    Real_IR_gray = torch.squeeze(
        .299 * Real_IR_norm[:, 0:1, :, :] + .587 * Real_IR_norm[:, 1:2, :, :] + .114 * Real_IR_norm[:, 2:3, :, :])
    Fake_IR_gray = torch.squeeze(
        .299 * Fake_IR_norm[:, 0:1, :, :] + .587 * Fake_IR_norm[:, 1:2, :, :] + .114 * Fake_IR_norm[:, 2:3, :, :])
    Fake_IR_flip_gray = torch.squeeze(
        .299 * Fake_IR_flip_norm[:, 0:1, :, :] + .587 * Fake_IR_flip_norm[:, 1:2, :, :] + .114 * Fake_IR_flip_norm[:,
                                                                                                 2:3, :, :])
    FG_area = torch.sum(FG_LL_mask + FG_LL_mask_flip)
    Fake_IR_FG_mean = (torch.sum(FG_LL_mask.mul(Fake_IR_gray)) + torch.sum(FG_LL_mask_flip.mul(Fake_IR_flip_gray))) / (
            FG_area + 1)
    Fake_IR_Fused = FG_LL_mask.mul(Fake_IR_gray) + FG_LL_mask_flip.mul(Fake_IR_flip_gray)
    Fake_IR_Fused_MaxValue = torch.max(Fake_IR_Fused)
    if torch.sum(RealIR_RoadMask) > 0:
        ######Adaptive luminance adjustment strategy: Adaptive scaling of luminance adjustment based on the mean value of the road area
        real_IR_Road_Mean = torch.sum(RealIR_RoadMask.mul(Real_IR_gray)) / torch.sum(RealIR_RoadMask)
        RB_Scale_Mean = real_IR_Road_Mean.detach() / (Fake_IR_FG_mean.detach() + 1e-6)
        real_IR_Road_MaxValue = torch.max(RealIR_RoadMask.mul(Real_IR_gray))
        #######Prevent the maximum value from crossing the boundary.
        RB_Scale_Max = real_IR_Road_MaxValue.detach() / (Fake_IR_Fused_MaxValue.detach() + 1e-6)
        RB_Scale = torch.min(RB_Scale_Mean, RB_Scale_Max)

    else:
        RB_Scale_Mean = torch.mean(Real_IR_gray) / (Fake_IR_FG_mean.detach() + 1e-6)
        real_IR_MaxValue = torch.max(Real_IR_gray)
        RB_Scale_Max = real_IR_MaxValue.detach() / (Fake_IR_Fused_MaxValue.detach() + 1e-6)
        RB_Scale = torch.min(RB_Scale_Mean, RB_Scale_Max)

    Fake_IR_masked_RB_norm = RB_Scale * (FG_LL_mask.mul(Fake_IR_gray)) + (FG_mask - FG_LL_mask).mul(Fake_IR_gray)
    Fake_IR_masked_flip_RB_norm = RB_Scale * (FG_LL_mask_flip.mul(Fake_IR_flip_gray)) + \
                                  (FG_mask_flip - FG_LL_mask_flip).mul(Fake_IR_flip_gray)

    Fake_IR_masked_RB = ((Fake_IR_masked_RB_norm - 0.5) * 2.0).mul(FG_mask)
    Fake_IR_masked_flip_RB = ((Fake_IR_masked_flip_RB_norm - 0.5) * 2.0).mul(FG_mask_flip)

    IR_com = (torch.ones_like(FG_mask) - FG_mask - FG_mask_flip).mul(Real_IR) + \
             Fake_IR_masked_RB.expand_as(Real_IR) + Fake_IR_masked_flip_RB.expand_as(Real_IR)

    return IR_com


def get_ROI_top_part_mask(input_mask, gpu_ids):
    h, w = input_mask.size()
    row_pos_array = torch.mm(torch.as_tensor(torch.arange(h).reshape((h, 1)), dtype=torch.float), torch.ones((1, h)))
    row_pos_array_masked = input_mask * (row_pos_array.cuda(gpu_ids))
    center_row_tensor = torch.sum(row_pos_array_masked, dim=0) / (torch.sum(input_mask, dim=0) + 1e-6)
    out_mask = torch.zeros_like(input_mask)

    for i in range(w):
        temp_row = int(center_row_tensor[i])
        if temp_row > 0:
            out_mask[:temp_row, i] = torch.ones(1).cuda(gpu_ids)

    mask_top_part = out_mask * input_mask

    return mask_top_part


######The conversion code between rgb and hsv images is derived from https://blog.csdn.net/Brikie/article/details/115086835.
def rgb_to_hsv(input_rgb):
    "input_rgb : 4D tensor."
    hue = torch.Tensor(input_rgb.shape[0], input_rgb.shape[2], input_rgb.shape[3]).to(input_rgb.device)

    hue[input_rgb[:, 2] == input_rgb.max(1)[0]] = 4.0 + (
            (input_rgb[:, 0] - input_rgb[:, 1]) / (input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8))[
        input_rgb[:, 2] == input_rgb.max(1)[0]]
    hue[input_rgb[:, 1] == input_rgb.max(1)[0]] = 2.0 + (
            (input_rgb[:, 2] - input_rgb[:, 0]) / (input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8))[
        input_rgb[:, 1] == input_rgb.max(1)[0]]
    hue[input_rgb[:, 0] == input_rgb.max(1)[0]] = (0.0 + (
            (input_rgb[:, 1] - input_rgb[:, 2]) / (input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8))[
        input_rgb[:, 0] == input_rgb.max(1)[0]]) % 6

    hue[input_rgb.min(1)[0] == input_rgb.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (input_rgb.max(1)[0] - input_rgb.min(1)[0]) / (input_rgb.max(1)[0] + 1e-8)
    saturation[input_rgb.max(1)[0] == 0] = 0

    value = input_rgb.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


def hsv_to_rgb(input_hsv):
    "input_hsv : 4D tensor."
    h, s, v = input_hsv[:, 0, :, :], input_hsv[:, 1, :, :], input_hsv[:, 2, :, :]
    # ###Treatment of out-of-bounds values
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb


def get_light_color(img_ref, img_test, pos, radius):
    h, w = img_ref.shape[-2:]
    patch_ref = img_ref[:, :, max(int(pos[0] - radius), 0):min(int(pos[0] + radius), h),
            max(int(pos[1] - radius), 0):min(int(pos[1] + radius), w)]
    patch_mean = patch_ref.mean(dim=[2, 3])[0]
    if patch_mean[0] - patch_mean[2] > 0:  # if not green
        if patch_mean[0] > patch_mean[1] * 1.5:  # if not orange
            color = Tensor([1, 0.2, 0]).to(img_ref.device)
        else:
            color = Tensor([1, 0.5, 0.1]).to(img_ref.device)
    else:
        color = Tensor([0.1, 0.9, 0.4]).to(img_ref.device)
    return color


def create_fake_TLight(img, mask_p):
    TLight_region = mask_p.mul(img)
    img_processed = TLight_region ** 7
    m = TLight_region.std(dim=1, keepdim=True) > (
                (TLight_region > 0) * TLight_region.std(dim=1, keepdim=True)).sum() / (
                    (TLight_region > 0).sum() + 1e-6)
    img_processed = img_processed * m.expand_as(img_processed)
    padsize = 5 // 2
    MaxPool_k5 = nn.MaxPool2d(5, stride=1, padding=padsize)
    for i in range(2):
        img_processed = MaxPool_k5(img_processed)
        img_processed = gaussian_blur(img_processed / (img_processed.max() + 1e-14), (11, 11), (5., 5.))
    img_processed = (img_processed / (img_processed.max() + 1e-14) + TLight_region*0.1).clamp(0, 1)
    fake = torch.zeros_like(img_processed).to(img.device)
    label_connect, num = measure.label((img_processed.mean(dim=1)>img_processed.mean() + img_processed.std()).cpu(), connectivity=2, background=0, return_num=True)
    for j in range(1, num + 1):
        "Since background index is 0, the num is num+1."
        temp_connect_mask = torch.where(torch.from_numpy(label_connect) == j, 1.0, 0.0).to(img.device)
        light_i = temp_connect_mask.expand_as(img_processed) * img_processed
        patch_max = light_i[0].flatten(1)[:, light_i[0].flatten(1).mean(dim=0)>0].max(dim=1)[0]
        patch_mean = light_i[0].flatten(1)[:, light_i[0].flatten(1).mean(dim=0)>0].mean(dim=1)
        patch_overlap = gaussian_blur(temp_connect_mask.expand_as(img_processed), (11, 11), (7., 7.))
        patch_overlap /= patch_overlap.max()
        # patch_overlap_neg = (1 - patch_overlap) * (patch_overlap>0)
        if patch_mean[0] - 1.5 * patch_mean[2] > 0:  # if red
            light_i = patch_overlap * light_i * 3
            light_i = light_i.clamp(0, 1)
        elif patch_mean[2] - 1.5 * patch_mean[0] > 0:  # if green
            light_i = patch_overlap * light_i * 3
            light_i = light_i.clamp(0, 1)
        else:
            light_i = 0
        fake += light_i
    return fake/(fake.max() + 1e-6)


def create_fake_Light(img, mask_p):
    fake = torch.zeros_like(img).to(img.device)
    b, c, h_, w_ = fake.shape
    label_connect, num = measure.label(mask_p.cpu(), connectivity=2, background=0, return_num=True)
    for j in range(1, num + 1):
        "Since background index is 0, the num is num+1."
        temp_connect_mask = torch.where(torch.from_numpy(label_connect) == j, 1.0, 0.0).to(img.device)
        h, w = temp_connect_mask.sum(dim=-2).max() + 1e-14, temp_connect_mask.sum(dim=-1).max()
        kernel_size = max(int(h * 2 + 1), 5), max(int(w * 2 + 1), 5)
        sigma = torch.tensor([min(h / 2, kernel_size[0]/3)]).to(img.device), torch.tensor([min(w / 2, kernel_size[1]/3)]).to(img.device)
        if w / h > 1.75:
            # Horizontal white streetlight from the top
            Light_region = mask_p.mul(torch.Tensor([1., 0.9, 0.85])[None, :, None, None].expand_as(img).to(mask_p.device))
            #drawn a bit lower
            temp = torch.zeros([b, c, h_+3, w_]).to(img.device)
            temp[:, :, 3:] = Light_region
            Light_region = temp[:, :, :-3]
            fake += gaussian_blur(Light_region, kernel_size, (1.6, 2))
        else:
            color = [1., 0.7, 0.05] if torch.rand(1)>0.5 else [1., 0.95, 0.95]
            Light_region = mask_p.mul(
                torch.Tensor(color)[None, :, None, None].expand_as(img).to(mask_p.device))
            fake += gaussian_blur(Light_region, kernel_size, sigma)
    img_processed = fake / fake.max() + img*mask_p
    return img_processed.clamp(0, 1)





def split_im(im, chunk_nb):
    chunk_size_w = im.shape[-1] // chunk_nb
    chunk_size_h = im.shape[-2] // chunk_nb
    return im.unfold(-2, chunk_size_h, chunk_size_h).unfold(-2, chunk_size_w, chunk_size_w).reshape(*im.shape[:2],
                                                                                                    chunk_nb ** 2,
                                                                                                    chunk_size_h,
                                                                                                    chunk_size_w)


# def get_light_color()

def detect_blob(image, method: Literal['LoG', 'DoG', 'DoH'] = 'LoG', min_radius=1, scale_blob=1, scale=3):
    h, w = image.shape
    image_process = torch.from_numpy(image - image.mean() - image.std()).clamp(0, 1).numpy()
    if method == 'LoG':
        blobs = blob_log(image_process, max_sigma=10, num_sigma=5, threshold=0.2)
        for i in range(scale):
            image_process = cv.pyrDown(image_process)
            np.concatenate((blobs, blob_log(image_process, max_sigma=50, num_sigma=10, threshold=0.1) * (i + 1) ** 2))
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    elif method == 'DoG':
        blobs = blob_dog(image_process, max_sigma=50, threshold=0.1)
        for i in range(scale):
            image_process = cv.pyrDown(image_process)
            np.concatenate((blobs, blob_dog(image_process, max_sigma=50, threshold=0.1) * (i + 1) ** 2))
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
    elif method == 'DoH':
        blobs = blob_doh(image_process, max_sigma=50, threshold=0.01)
        for i in range(scale):
            image_process = cv.pyrDown(image_process)
            np.concatenate((blobs, blob_doh(image_process, max_sigma=50, threshold=0.01) * (i + 1) ** 2))
    else:
        raise TypeError
    res = torch.zeros([1, 3, h, w]).to(image.device)
    for blob in blobs:
        if blob[2] > min_radius:
            rad = int(blob[2]) * scale_blob
            rad_k = rad * 2 + 1
            ker = get_gaussian_kernel2d((rad_k, rad_k), (blob[2] ** 0.5, blob[2] ** 0.5)).squeeze(0)
            ker = ker.clamp(ker.max() / 15, 1)
            ker = ker - ker.min()
            x0, x1 = int(max(0, np.ceil(blob[1] - ker.shape[-1] / 2))), int(
                min(w, np.ceil(blob[1] + ker.shape[-1] / 2)))
            y0, y1 = int(max(0, np.ceil(blob[0] - ker.shape[-2] / 2))), int(
                min(h, np.ceil(blob[0] + ker.shape[-2] / 2)))
            y0_ker, y1_ker = 0 if y0 > 0 else rad_k - (y1 - y0), rad_k if y1 < h else int(y1 - y0)
            x0_ker, x1_ker = 0 if x0 > 0 else rad_k - (x1 - x0), rad_k if x1 < w else int(x1 - x0)
            res[:, :, y0:y1, x0:x1] += ker[y0_ker:y1_ker, x0_ker:x1_ker]
    return res == 0

# def detect_blob(image, min_radius=2, scale_blob=2, scale=3):
#     h, w = image.shape
#     filters = torch.cat([get_gaussian_kernel2d((21, 21), (sigma/2, sigma/2)) for sigma in
#                torch.arange(1, 6, step=1)]).unsqueeze(1)
#     image_process = torch.from_numpy(image - image.mean() - 3*image.std()).clamp(0, 1)
#     image_ = ImageTensor(image_process, normalize=True)
#     conv = Conv2d(1, 5, (21, 21), stride=1, padding=10, dilation=1)
#     conv.weights = filters
#     softmax = Softmax2d()
#     filtered = []
#     for i in range(scale):
#         filtered.append(F.interpolate(conv(image_), (h, w)))
#         image_ = image_.pyrDown()
#     filtered = torch.cat(filtered)
#     filtered, scale = torch.max(filtered, dim=0, keepdim=True)
#     filtered = softmax(filtered)
#     filtered, sigmas = torch.max(filtered, dim=1, keepdim=True)
#     filtered = filtered * ImageTensor(image_process)
#     f = filtered <= filtered.max() * 0.9
#     return f
