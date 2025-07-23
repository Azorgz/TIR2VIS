import numpy as np
import torch
from torch import nn
from pytorch_msssim import SSIM
import torch.nn.functional as F
from torchvision.transforms.v2.functional import gaussian_blur

from ImagesCameras import ImageTensor
from models.networks import Vgg16, Get_gradmag_gray, RGBuvHistBlock
from models.utils_fct import get_ROI_top_part_mask, GetFeaMatrixCenter, bhw_to_onehot, ClsMeanPixelValue, \
    getLightDarkRegionMean, RefineLightMask, split_im


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return (1 - super(SSIM_Loss, self).forward(img1, img2))


# Defines the total variation (TV) loss, which encourages spatial smoothness in the generated image.
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def TrafLighCorlLoss(real_IR, fake_vis, IR_mask, real_vis, vis_Light_mask, HL_Mask_ori, gpu_ids):
    "Traffic light color loss: The color distribution of the traffic lights in the fake visible image is encouraged to be consistent "
    "with the real image."

    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    vis_HL_Mask = -max_pool_k5(-max_pool_k5(HL_Mask_ori))
    if torch.sum(vis_HL_Mask) > 0:
        IR_seg_mask_2D = torch.squeeze(IR_mask.argmax(dim=1))
        # h, w = IR_seg_mask_2D.size()
        IR_Light_mask = torch.where(IR_seg_mask_2D == 6.0, 1., 0.)
        if torch.sum(IR_Light_mask) > 100:
            IR_Light_top_mask = get_ROI_top_part_mask(IR_Light_mask, gpu_ids)
            vis_Light_top_mask = vis_Light_mask[0, :, :]
            IR_Light_bottom_mask = IR_Light_mask - IR_Light_top_mask
            vis_Light_bottom_mask = vis_Light_mask[1, :, :]
            Real_IR_norm = (real_IR + 1.0) * 0.5
            Fake_vis_norm = torch.squeeze((fake_vis + 1.0) * 0.5)
            Real_vis_norm = torch.squeeze((real_vis + 1.0) * 0.5)
            Real_IR_gray = torch.squeeze(
                .299 * Real_IR_norm[:, 0:1, :, :] + .587 * Real_IR_norm[:, 1:2, :, :] + .114 * Real_IR_norm[:, 2:3, :,
                                                                                               :])
            IR_gray_light = Real_IR_gray * IR_Light_mask
            IR_gray_light_mean = torch.sum(IR_gray_light) / torch.sum(IR_Light_mask)
            IR_HL_mask = torch.zeros_like(IR_seg_mask_2D)
            IR_HL_mask = torch.where(IR_gray_light > IR_gray_light_mean, torch.ones_like(IR_seg_mask_2D),
                                     torch.zeros_like(IR_seg_mask_2D))
            IR_top_HL_mask = IR_HL_mask * IR_Light_top_mask
            IR_bottom_HL_mask = IR_HL_mask * IR_Light_bottom_mask
            vis_top_HL_mask = vis_HL_Mask[0, 0, :, :] * vis_Light_top_mask
            vis_bottom_HL_mask = vis_HL_Mask[0, 0, :, :] * vis_Light_bottom_mask
            HL_top_idx = torch.sum(IR_top_HL_mask) * torch.sum(vis_top_HL_mask)
            HL_bottom_idx = torch.sum(IR_bottom_HL_mask) * torch.sum(vis_bottom_HL_mask)
            if HL_top_idx > 0:
                fake_vis_top_masked = Fake_vis_norm.mul(IR_top_HL_mask.expand_as(Fake_vis_norm))
                fake_vis_top_HL_Light_mean = torch.sum(fake_vis_top_masked, dim=(1, 2), keepdim=True) / torch.sum(
                    IR_top_HL_mask)
                real_vis_top_masked = Real_vis_norm.mul(vis_top_HL_mask.expand_as(Real_vis_norm))
                real_vis_top_HL_Light_mean = torch.sum(real_vis_top_masked, dim=(1, 2), keepdim=True) / torch.sum(
                    vis_top_HL_mask)
                HL_top_loss = torch.sqrt(
                    torch.sum((real_vis_top_HL_Light_mean.detach() - fake_vis_top_HL_Light_mean) ** 2))
            else:
                HL_top_loss = torch.zeros(1).cuda(gpu_ids)

            if HL_bottom_idx > 0:
                fake_vis_bottom_masked = Fake_vis_norm.mul(IR_bottom_HL_mask.expand_as(Fake_vis_norm))
                fake_vis_bottom_HL_Light_mean = torch.sum(fake_vis_bottom_masked, dim=(1, 2), keepdim=True) / torch.sum(
                    IR_bottom_HL_mask)
                real_vis_bottom_masked = Real_vis_norm.mul(vis_bottom_HL_mask.expand_as(Real_vis_norm))
                real_vis_bottom_HL_Light_mean = torch.sum(real_vis_bottom_masked, dim=(1, 2), keepdim=True) / torch.sum(
                    vis_bottom_HL_mask)
                # HL_bottom_loss = torch.sqrt(torch.sum((real_vis_bottom_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                bottom_loss_sim = torch.sqrt(
                    torch.sum((real_vis_bottom_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                if HL_top_idx > 0:
                    bottom_loss_var = torch.sqrt(
                        torch.sum((real_vis_top_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                    Norm_factor = torch.min(bottom_loss_sim.detach(), bottom_loss_var)
                    HL_bottom_loss = bottom_loss_sim / (Norm_factor + 0.05)
                else:
                    HL_bottom_loss = bottom_loss_sim

            else:
                HL_bottom_loss = torch.zeros(1).cuda(gpu_ids)

            out_losses = HL_top_loss + HL_bottom_loss
        else:
            out_losses = torch.zeros(1).cuda(gpu_ids)
    else:
        out_losses = torch.zeros(1).cuda(gpu_ids)

    return out_losses


def StruGradAligLoss(real_IR_edgemap, fake_vis_gradmap, sqrt_patch_num, gradient_th):
    "SGA Loss. The ratio of the gradient at the edge location to the maximum gradient in "
    "its neighborhood is encouraged to be greater than a given threshold."

    b, c, h, w = fake_vis_gradmap.size()
    # patch_num = sqrt_patch_num * sqrt_patch_num
    AAP_module = nn.AdaptiveAvgPool2d(sqrt_patch_num)
    real_IR_edgemap_pooling = AAP_module(real_IR_edgemap.expand_as(fake_vis_gradmap))
    if torch.sum(real_IR_edgemap) > 0:
        pooling_array = real_IR_edgemap_pooling[0].detach().cpu().numpy()
        h_nonzero, w_nonzero = np.nonzero(pooling_array[0])
        patch_idx_rand = np.random.randint(0, len(h_nonzero))
        patch_idx_x = h_nonzero[patch_idx_rand]
        patch_idx_y = w_nonzero[patch_idx_rand]
        crop_size = h // sqrt_patch_num
        pos_list = list(range(0, h, crop_size))

        pos_h = pos_list[patch_idx_x]
        pos_w = pos_list[patch_idx_y]
        # rand_patch = self.Tensor(b, c, crop_size, crop_size)
        rand_edgemap_patch = real_IR_edgemap[:, :, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
        rand_gradmap_patch = fake_vis_gradmap[:, :, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]

        sum_edge_pixels = torch.sum(rand_edgemap_patch) + 1
        # print('Sum_edge_pixels of IR edge map is: ', sum_edge_pixels.detach().cpu().numpy())
        fake_grad_norm = rand_gradmap_patch / torch.max(rand_gradmap_patch)
        losses = (torch.sum(F.relu(gradient_th * rand_edgemap_patch - fake_grad_norm))) / sum_edge_pixels
    else:
        losses = 0

    return losses


def compute_vgg_loss(img, target, gpu_ids=[]):
    # img_vgg = vgg_preprocess(img)
    # target_vgg = vgg_preprocess(target)
    vgg = Vgg16(gpu_ids)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    loss_mse = torch.nn.MSELoss()
    img_fea = vgg(img)
    target_fea = vgg(target)
    content_loss = 0.0
    for j in range(4):
        content_loss += loss_mse(img_fea[j], target_fea[j])

    return content_loss * 0.25


def ClsACALoss(real_vis_fea, cls_mask_real, fake_vis_fea, cls_mask_fake, fea_dim, cluster_num, max_iter, gpu_ids):
    "Calculating adaptive collaborative attention loss for a single class."

    real_fea_cls_masked = real_vis_fea.mul(cls_mask_real.expand_as(real_vis_fea))
    real_fea_cls_matrix = (real_fea_cls_masked.view(fea_dim, -1)).t()  ### N * c 256x4096
    nonZeroRows = torch.abs(real_fea_cls_matrix).sum(dim=1) > 0
    real_fea_cls_matrix = real_fea_cls_matrix[nonZeroRows]
    cls_cluster_center_array = GetFeaMatrixCenter(real_fea_cls_matrix, cluster_num, max_iter, gpu_ids)  ###Cn * c
    cls_center_fea_norm = F.normalize(cls_cluster_center_array, p=2, dim=1)
    real_fea_cls_matrix_norm = F.normalize(real_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_real = torch.mm(real_fea_cls_matrix_norm, cls_center_fea_norm.t())  ### Np * Cn
    cls_sim_map_max_real = torch.max(cls_sim_map_real, dim=1)
    cls_fea_sim_mean_real = torch.mean(cls_sim_map_max_real[0])
    sim_map_clustermax_real = torch.max(cls_sim_map_real, dim=0)
    fea_sim_clustermean_real = torch.mean(sim_map_clustermax_real[0])

    fake_fea_cls_masked = fake_vis_fea.mul(cls_mask_fake.expand_as(fake_vis_fea))
    fake_fea_cls_matrix = (fake_fea_cls_masked.view(fea_dim, -1)).t()  ### N * c
    fake_fea_cls_matrix_norm = F.normalize(fake_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_fake = torch.mm(fake_fea_cls_matrix_norm, cls_center_fea_norm.t())  ### N * Cn
    cls_sim_map_max_fake = torch.max(cls_sim_map_fake, dim=1)
    cls_fea_sim_mean_fake = torch.sum(cls_sim_map_max_fake[0]) / torch.sum(cls_mask_fake.detach())
    sim_map_clustermax_fake = torch.max(cls_sim_map_fake, dim=0)
    fea_sim_clustermean_fake = torch.mean(sim_map_clustermax_fake[0])
    loss_cls_sim = F.relu(0.9 * cls_fea_sim_mean_real.detach() - cls_fea_sim_mean_fake)
    loss_cls_div = F.relu(0.9 * fea_sim_clustermean_real.detach() - fea_sim_clustermean_fake)
    losses = loss_cls_sim + loss_cls_div

    return losses


def AdaColAttLoss(real_vis_mask, real_vis_fea, fake_vis_mask, fake_vis_fea, cluster_num, max_iter, gpu_ids):
    "Adaptive Collaborative Attention Loss."

    _, c, h, w = real_vis_fea.size()
    real_vis_mask_resize = F.interpolate(real_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')
    fake_vis_mask_resize = F.interpolate(fake_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')

    Light_mask_real = torch.where(real_vis_mask_resize == 6.0, torch.ones_like(real_vis_mask_resize),
                                  torch.zeros_like(real_vis_mask_resize))
    Sign_mask_real = torch.where(real_vis_mask_resize == 7.0, torch.ones_like(real_vis_mask_resize),
                                 torch.zeros_like(real_vis_mask_resize))
    Person_mask_real = torch.where(real_vis_mask_resize == 11.0, torch.ones_like(real_vis_mask_resize),
                                   torch.zeros_like(real_vis_mask_resize))
    Vehicle_mask_real = torch.where((real_vis_mask_resize > 12.0) & (real_vis_mask_resize < 17.0),
                                    torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Motor_mask_real = torch.where(real_vis_mask_resize == 17.0, torch.ones_like(real_vis_mask_resize),
                                  torch.zeros_like(real_vis_mask_resize))

    Light_mask_fake = torch.where(fake_vis_mask_resize == 6.0, torch.ones_like(fake_vis_mask_resize),
                                  torch.zeros_like(fake_vis_mask_resize))
    Sign_mask_fake = torch.where(fake_vis_mask_resize == 7.0, torch.ones_like(fake_vis_mask_resize),
                                 torch.zeros_like(fake_vis_mask_resize))
    Person_mask_fake = torch.where(fake_vis_mask_resize == 11.0, torch.ones_like(fake_vis_mask_resize),
                                   torch.zeros_like(fake_vis_mask_resize))
    Vehicle_mask_fake = torch.where((fake_vis_mask_resize > 12.0) & (fake_vis_mask_resize < 17.0),
                                    torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Motor_mask_fake = torch.where(fake_vis_mask_resize == 17.0, torch.ones_like(fake_vis_mask_resize),
                                  torch.zeros_like(fake_vis_mask_resize))

    if (torch.sum(Light_mask_real) > cluster_num) & (torch.sum(Light_mask_fake) > cluster_num):
        loss_light = ClsACALoss(real_vis_fea, Light_mask_real, fake_vis_fea, Light_mask_fake, c, cluster_num, max_iter,
                                gpu_ids)
        idx_light = 1.0
    else:
        loss_light = 0.0
        idx_light = 0.0

    if (torch.sum(Sign_mask_real) > cluster_num) & (torch.sum(Sign_mask_fake) > cluster_num):
        loss_sign = ClsACALoss(real_vis_fea, Sign_mask_real, fake_vis_fea, Sign_mask_fake, c, cluster_num, max_iter,
                               gpu_ids)
        idx_sign = 1.0
    else:
        loss_sign = 0.0
        idx_sign = 0.0

    if (torch.sum(Person_mask_real) > cluster_num) & (torch.sum(Person_mask_fake) > cluster_num):
        loss_person = ClsACALoss(real_vis_fea, Person_mask_real, fake_vis_fea, Person_mask_fake, c, cluster_num,
                                 max_iter, gpu_ids)
        idx_person = 1.0
    else:
        loss_person = 0.0
        idx_person = 0.0

    if (torch.sum(Vehicle_mask_real) > cluster_num) & (torch.sum(Vehicle_mask_fake) > cluster_num):
        loss_vehicle = ClsACALoss(real_vis_fea, Vehicle_mask_real, fake_vis_fea, Vehicle_mask_fake, c, cluster_num,
                                  max_iter, gpu_ids)
        idx_vehicle = 1.0
    else:
        loss_vehicle = 0.0
        idx_vehicle = 0.0

    if (torch.sum(Motor_mask_real) > cluster_num) & (torch.sum(Motor_mask_fake) > cluster_num):
        loss_motor = ClsACALoss(real_vis_fea, Motor_mask_real, fake_vis_fea, Motor_mask_fake, c, cluster_num, max_iter,
                                gpu_ids)
        idx_motor = 1.0
    else:
        loss_motor = 0.0
        idx_motor = 0.0

    obj_cls_num = idx_light + idx_sign + idx_person + idx_vehicle + idx_motor
    # obj_cls_num = idx_sign + idx_person + idx_vehicle + idx_motor
    if obj_cls_num > 0:
        losses = (loss_light + loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
        # losses = (loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
    else:
        losses = 0.0

    return losses


def SemEdgeLoss(seg_tensor, GT_mask, num_classes, gpu_ids):
    "Encourage semantic edge prediction consistent with GT."

    sm = torch.nn.Softmax(dim=1)
    pred_sm = sm(seg_tensor)
    GT2onehot = bhw_to_onehot(GT_mask, num_classes + 1, gpu_ids)
    AvgPool_k3 = nn.AvgPool2d(3, stride=1, padding=1)
    pred_semedge = torch.abs(pred_sm - AvgPool_k3(pred_sm))
    GT_semedge = torch.abs(GT2onehot - AvgPool_k3(GT2onehot))
    if torch.sum(GT_semedge) > 0:
        losses = torch.sum(torch.abs(GT_semedge.detach() - pred_semedge)) / torch.sum(GT_semedge.detach())
    else:
        losses = 0.0

    return losses


def FakeIRPersonLossv2(Seg_mask, fake_IR, real_vis, gpu_ids=[]):
    "Temperature regularization term: Encouraging the min value of the pedestrian region in the fake IR image "
    "to be larger than the mean value of the road region. "

    # b, c, h, w = fake_IR.size()

    b, c, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    # real_mask_resize = F.interpolate(real_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = GT_mask_resize[0]
    person_mask = torch.zeros_like(GT_mask_resize)
    person_mask = torch.where(GT_mask_resize == 11, torch.ones_like(GT_mask_resize), torch.zeros_like(GT_mask_resize))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    # real_img_norm = (real_vis + 1.0) * 0.5
    fake_IR_gray = .299 * fake_img_norm[:, 0:1, :, :] + .587 * fake_img_norm[:, 1:2, :, :] + .114 * fake_img_norm[:,
                                                                                                    2:3, :, :]
    fake_mean_fea, fake_cls_tensor, _ = ClsMeanPixelValue(fake_IR_gray, Seg_mask.detach(), 19, gpu_ids)
    if (fake_cls_tensor[11, :] * fake_cls_tensor[0, :]) > 0:
        person_region = (person_mask.expand_as(fake_IR_gray)).mul(fake_IR_gray)
        non_person_mask = torch.ones_like(person_mask) - person_mask
        person_region_padding1 = person_region + non_person_mask  ####To get person region min value
        road_mean_value = (fake_mean_fea[0, :]).detach()
        person_min_value = torch.min(person_region_padding1)
        person_dis_loss = F.relu(road_mean_value - person_min_value) / (road_mean_value + 1e-4)
    else:
        person_dis_loss = 0.0

    return person_dis_loss


def CarIntraClsVarLoss(input_IR, fake_vis, SegMask, num_class, gpu_ids=[]):
    "Encouraging intra-class feature variability in foreground categories in IR images."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = fake_vis.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]

    IR_gray = torch.squeeze(
        .299 * input_IR[:, 0:1, :, :] + .587 * input_IR[:, 1:2, :, :] + .114 * input_IR[:, 2:3, :, :])

    mask_intensity_low = torch.where(IR_gray < 0.3, torch.ones_like(IR_gray), torch.zeros_like(IR_gray))
    mask_intensity_high = torch.where(IR_gray > 0.6, torch.ones_like(IR_gray), torch.zeros_like(IR_gray))
    out_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)

    if b == 1:
        for i in range(13, 16):
            temp_tensor = torch.zeros_like(seg_mask)
            temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
            temp_tensor_low = temp_tensor.mul(mask_intensity_low)
            temp_tensor_high = temp_tensor.mul(mask_intensity_high)
            # temp_tensor = att_maps[0, i, :, :]
            if (torch.sum(temp_tensor_low)).item() > 0:
                # print((torch.sum(temp_tensor)).item())
                out_cls_tensor[i, 0] = 1.0
                # out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                fea_map_low = (temp_tensor_low.detach().expand_as(fake_vis)).mul(fake_vis)
                fea_low = (torch.squeeze(GAP(fea_map_low) * h * w)) / torch.sum(temp_tensor_low)  # b * c * 1 * 1

                out_tensor[i, 0] = torch.mean(fea_low)
    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    if torch.sum(out_cls_tensor).item() > 0:
        out_loss = F.relu((torch.sum(out_cls_tensor.mul(out_tensor)) / torch.sum(out_cls_tensor)) - 0.3)
    else:
        out_loss = torch.zeros(1).cuda(gpu_ids)

    return out_loss


def CondGradRepaLoss(fake_img, fake_mask, real_IR, gpu_ids=[]):
    "Conditional Gradient Repair loss for background categories. fake_img: fake vis image. fake_mask: IR seg mask."

    ###Conditional Gradient Repair loss for background categories
    b, _, h, w = fake_img.size()
    _, _, seg_h, seg_w = fake_mask.size()
    fake_mask_resize = F.interpolate(fake_mask.float(), size=[h, w], mode='nearest')

    seg_mask_fake = fake_mask_resize[0]
    IR_bkg_mask = torch.where(seg_mask_fake < 11.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_UC_mask = torch.where(seg_mask_fake == 255.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_bkg_fuse_mask = IR_bkg_mask + IR_UC_mask
    getgradmap = Get_gradmag_gray()
    IR_grad = getgradmap(real_IR)
    fake_vis_grad = getgradmap(fake_img)
    IR_grad_bkg = IR_grad.mul(IR_bkg_fuse_mask.expand_as(IR_grad))
    vis_grad_bkg = fake_vis_grad.mul(IR_bkg_fuse_mask.expand_as(fake_vis_grad))
    IR_grad_bkg_sum = torch.sum(IR_grad_bkg)

    if IR_grad_bkg_sum > 0:
        # bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg.detach() - vis_grad_bkg)) / IR_grad_bkg_sum.detach()

        IR_grad_bkg_mean = IR_grad_bkg_sum / torch.sum(IR_bkg_fuse_mask)
        IR_grad_bkg_high_mask = torch.zeros_like(IR_grad_bkg)
        IR_grad_bkg_high_mask = torch.where(IR_grad_bkg > IR_grad_bkg_mean, torch.ones_like(IR_grad_bkg),
                                            torch.zeros_like(IR_grad_bkg))
        IR_grad_bkg_high = IR_grad_bkg_high_mask.mul(IR_grad_bkg)
        vis_grad_bkg_high = IR_grad_bkg_high_mask.mul(vis_grad_bkg)
        IR_grad_bkg_high_sum = torch.sum(IR_grad_bkg_high)
        if IR_grad_bkg_high_sum > 0:
            bkg_EC_loss = torch.sum(
                F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high)) / IR_grad_bkg_high_sum.detach()
        else:
            bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)
    else:
        bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)

    return bkg_EC_loss


def TrafLighLumiLoss(fake_img, fake_mask, real_IR, gpu_ids=[]):
    "Traffic Light Luminance Loss. fake_img: fake vis image. fake_mask: IR seg mask. real_mask: Vis seg mask."
    _, _, h, w = fake_img.size()
    _, _, seg_h, seg_w = fake_mask.size()
    fake_mask_resize = F.interpolate(fake_mask.float(), size=[h, w], mode='nearest')

    fake_img_norm = (fake_img + 1.0) * 0.5
    real_IR_norm = (real_IR + 1.0) * 0.5

    fake_vis_Light_DR_Mean, fake_vis_Light_area, fake_vis_Light_BR_Min, _ = \
        getLightDarkRegionMean(6.0, fake_img_norm, fake_mask_resize, real_IR_norm.detach(), gpu_ids)

    if fake_vis_Light_area > 100:
        losses = F.relu(fake_vis_Light_DR_Mean - fake_vis_Light_BR_Min) / (fake_vis_Light_BR_Min.detach() + 1e-6)
    else:
        losses = torch.zeros(1).cuda(gpu_ids)

    return losses


def MaskedCGRLoss(input_mask, real_IR, fake_vis, gpu_ids):
    "Conditional Gradient Repairing loss for input binary mask."

    getgradmap = Get_gradmag_gray()
    IR_grad = getgradmap(real_IR)
    fake_vis_grad = getgradmap(fake_vis)
    IR_grad_bkg = IR_grad.mul(input_mask)
    vis_grad_bkg = fake_vis_grad.mul(input_mask)
    IR_grad_bkg_sum = torch.sum(IR_grad_bkg)

    if IR_grad_bkg_sum > 0:
        # bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg.detach() - vis_grad_bkg)) / IR_grad_bkg_sum.detach()

        IR_grad_bkg_mean = IR_grad_bkg_sum / torch.sum(input_mask)
        IR_grad_bkg_high_mask = torch.zeros_like(IR_grad_bkg)
        IR_grad_bkg_high_mask = torch.where(IR_grad_bkg > IR_grad_bkg_mean, torch.ones_like(IR_grad_bkg),
                                            torch.zeros_like(IR_grad_bkg))
        IR_grad_bkg_high = IR_grad_bkg_high_mask.mul(IR_grad_bkg)
        vis_grad_bkg_high = IR_grad_bkg_high_mask.mul(vis_grad_bkg)
        IR_grad_bkg_high_sum = torch.sum(IR_grad_bkg_high)
        if IR_grad_bkg_high_sum > 0:
            losses = torch.sum(F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high)) / IR_grad_bkg_high_sum.detach()
        else:
            losses = torch.zeros(1).cuda(gpu_ids)
    else:
        losses = torch.zeros(1).cuda(gpu_ids)

    return losses


def ComIRCGRLoss(FG_mask, FG_mask_flip, ori_Seg_GT, real_IR, fake_vis, gpu_ids):
    "Conditional Gradient Repairing loss for composite IR."
    _, _, h, w = fake_vis.size()
    _, _, seg_h, seg_w = ori_Seg_GT.size()
    IR_mask_resize = F.interpolate(ori_Seg_GT.float(), size=[h, w], mode='nearest')
    seg_mask_fake = torch.squeeze(IR_mask_resize)  #h * w

    FG_mask_fused_4d = FG_mask + FG_mask_flip
    FG_mask_fused = FG_mask_fused_4d[0, 0, :, :]
    IR_bkg_mask = torch.where(seg_mask_fake < 11.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_UC_mask = torch.where(seg_mask_fake == 255.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_bkg_ori_mask = IR_bkg_mask + IR_UC_mask
    ComIR_bkg_mask = IR_bkg_ori_mask - FG_mask_fused.mul(IR_bkg_ori_mask)
    losses = MaskedCGRLoss(ComIR_bkg_mask, real_IR, fake_vis, gpu_ids)

    return losses


def BiasCorrLoss(Seg_mask, fake_IR, real_vis, rec_vis, real_vis_edgemap, gpu_ids=[]):
    """
    Seg_mask :: GT mask
    fake_IR :: fake IR image
    """
    "Bias correction loss includes the artifact bias correction loss and the color bias correction loss. "
    "The category index for streetlights is defined as 12."

    _, _, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = torch.squeeze(GT_mask_resize[0])

    light_mask_ori = torch.where(GT_mask == 6.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    veg_mask = torch.where(GT_mask == 8.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    SLight_mask_ori = torch.where(GT_mask == 12.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    light_mask = RefineLightMask(GT_mask, real_vis, gpu_ids)

    fake_img_norm = (fake_IR + 1.0) * 0.5
    real_img_norm = (real_vis + 1.0) * 0.5
    fake_IR_gray = torch.squeeze(
        .299 * fake_img_norm[:, 0:1, :, :] + .587 * fake_img_norm[:, 1:2, :, :] + .114 * fake_img_norm[:, 2:3, :, :])
    real_vis_gray = torch.squeeze(
        .299 * real_img_norm[:, 0:1, :, :] + .587 * real_img_norm[:, 1:2, :, :] + .114 * real_img_norm[:, 2:3, :, :])

    ###########Artifact bias correction loss
    ####Street light luminance adjustment loss
    "Excluding the noise at the periphery and the street light area less than 25."
    max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    SLight_mask = torch.squeeze(-max_pool_k3(- SLight_mask_ori.expand(1, 1, h, w)))

    if torch.sum(SLight_mask) > 25:
        real_vis_SLight_region = SLight_mask.mul(real_vis_gray)
        real_vis_SLight_mean = torch.sum(real_vis_SLight_region) / torch.sum(SLight_mask)
        SLight_high_mask = torch.where(real_vis_SLight_region > real_vis_SLight_mean, torch.ones_like(GT_mask),
                                       torch.zeros_like(GT_mask))
        fake_IR_SLight_region_high = SLight_high_mask.mul(fake_IR_gray) + (
                torch.ones_like(SLight_high_mask) - SLight_high_mask)

        if torch.sum(veg_mask) > 0:
            fake_IR_veg_region = veg_mask.mul(fake_IR_gray)
            fake_IR_veg_mean = torch.sum(fake_IR_veg_region) / torch.sum(veg_mask)
            fake_IR_veg_max = torch.max(fake_IR_veg_region)
            "Avoid the maximum value of the brightness of the vegetation area is too high."
            SLight_loss = F.relu(fake_IR_veg_mean.detach() + 0.25 - torch.min(fake_IR_SLight_region_high))
        else:
            SLight_loss = F.relu(0.7 - torch.min(fake_IR_SLight_region_high))
    else:
        SLight_loss = 0.0

    ########Light region SGA loss
    light_mask_all = light_mask_ori + SLight_mask_ori
    if torch.sum(light_mask_all) > 100:
        real_vis_EM = torch.squeeze(real_vis_edgemap)
        gradmag_com = Get_gradmag_gray()
        fake_IR_GM = torch.squeeze(gradmag_com(fake_IR))

        EM_masked = light_mask_all.mul(real_vis_EM)
        GM_masked = light_mask_all.mul(fake_IR_GM)

        sum_edge_pixels = torch.sum(EM_masked)
        if sum_edge_pixels > 0:
            fake_grad_norm = GM_masked / (torch.max(GM_masked) + 1e-4)
            loss_sga_light = 0.5 * (torch.sum(F.relu(0.8 * EM_masked - fake_grad_norm))) / sum_edge_pixels
        else:
            loss_sga_light = 0.0
    else:
        loss_sga_light = 0.0
    #################

    ####Traffic light luminance adjustment loss
    if torch.sum(light_mask) > 100:

        real_vis_light_region = light_mask.mul(real_vis_gray)
        real_vis_light_mean = torch.sum(real_vis_light_region) / torch.sum(light_mask)
        real_vis_light_region_submean = light_mask.mul(real_vis_light_region - real_vis_light_mean)
        real_vis_light_region_norm2 = torch.sqrt(torch.sum(real_vis_light_region_submean ** 2))
        real_vis_light_norm = real_vis_light_region_submean / (real_vis_light_region_norm2 + 1e-4)
        light_high_mask = torch.where(real_vis_light_region > real_vis_light_mean, torch.ones_like(GT_mask),
                                      torch.zeros_like(GT_mask))

        high_area_ratio = torch.sum(light_high_mask) / torch.sum(light_mask)

        if high_area_ratio > 0.1:
            fake_IR_light_region = light_mask.mul(fake_IR_gray)
            fake_IR_light_mean = torch.sum(fake_IR_light_region) / torch.sum(light_mask)
            fake_IR_light_region_submean = light_mask.mul(fake_IR_light_region - fake_IR_light_mean)
            fake_IR_light_region_norm2 = torch.sqrt(torch.sum(fake_IR_light_region_submean ** 2))
            fake_IR_light_norm = fake_IR_light_region_submean / (fake_IR_light_region_norm2 + 1e-4)

            TLight_loss = F.relu(0.8 - torch.sum(fake_IR_light_norm.mul(real_vis_light_norm.detach())))
        else:
            TLight_loss = 0.0
    else:
        TLight_loss = 0.0

    ABC_losses = TLight_loss + SLight_loss + loss_sga_light

    ##########Color bias correction loss
    ####Traffic sign reconstruction loss
    sign_mask = torch.where(GT_mask == 7.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    if torch.sum(sign_mask) > 10:
        sign_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, sign_mask, 3)
    else:
        sign_rec_loss = 0.0
    ####Traffic light reconstruction loss
    if torch.sum(light_mask) > 10:
        light_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, light_mask, 3)
    else:
        light_rec_loss = 0.0
    ####Motorcycle reconstruction loss
    motorcycle_mask = torch.zeros_like(GT_mask)
    motorcycle_mask = torch.where(GT_mask == 17.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    if torch.sum(motorcycle_mask) > 10:
        motorcycle_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, motorcycle_mask, 3)
    else:
        motorcycle_rec_loss = 0.0

    CBC_losses = sign_rec_loss + light_rec_loss + motorcycle_rec_loss

    out_losses = ABC_losses + CBC_losses

    return out_losses


def PixelConsistencyLoss(inputs_img, GT_img, ROI_mask, ssim_winsize):
    "Pixel-wise Consistency Loss. inputs_img and GT_img are 4D tensors range [-1, 1], while ROI_mask is a 2D tensor."
    input_masked = inputs_img.mul(ROI_mask.expand_as(inputs_img))
    GT_masked = GT_img.mul(ROI_mask.expand_as(GT_img))
    # print(len(ROI_mask.size()))
    if len(ROI_mask.size()) == 4:
        _, _, h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask[0, 0, :, :])
    elif len(ROI_mask.size()) == 3:
        _, h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask[0, :, :])
    else:
        h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask)

    criterionSSIM = SSIM_Loss(win_size=ssim_winsize, data_range=1.0, size_average=True, channel=3)
    criterionL1 = torch.nn.SmoothL1Loss()
    lambda_L1 = 10.0
    if area_ROI > 0:
        losses = ((h * w) / area_ROI) * (lambda_L1 * criterionL1(input_masked, GT_masked.detach()) + \
                                         criterionSSIM((input_masked + 1) / 2, (GT_masked.detach() + 1) / 2))
    else:
        losses = 0.0

    return losses


# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r, f: torch.mean((r - f + y) ** 2)
    losses = [loss(r, f) for r, f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real) + 1))
    multipliers[-1] += 1
    losses = [m * l for m, l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))


def HistogramLoss(fake_im, real_color, GT_seg):
    chunk_nb = 4
    size = fake_im.shape[-1]//chunk_nb
    hist_block = RGBuvHistBlock(insz=size, h=64,
                 intensity_scale=True,
                 method='inverse-quadratic',
                 device=fake_im.device)
    histogram_loss = lambda target_hist, input_hist: (1 / np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2))+1e-6)) /
                      input_hist.shape[0])
    losses = []
    # Cars color
    fake_im = fake_im*0.5 + 0.5
    real_color = real_color*0.5 + 0.5
    if GT_seg is None:
        chunk_nb = 1
        size = fake_im.shape[-1] // chunk_nb
        hist_block = RGBuvHistBlock(insz=size, h=16,
                                    intensity_scale=True,
                                    method='inverse-quadratic',
                                    device=fake_im.device)
        hist_fake = hist_block(fake_im)
        hist_real = hist_block(real_color)
        losses.append(histogram_loss(hist_real, hist_fake))
    else:
        mask = torch.where(GT_seg == 13, 1., 0.)
        if torch.sum(mask) > 0:
            fake = split_im(fake_im * mask, chunk_nb)
            real = split_im(real_color * mask, chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss)
        # Trucks color
        mask = torch.where(GT_seg == 14, 1., 0.)
        if torch.sum(mask) > 0:
            fake = split_im(fake_im * mask, chunk_nb)
            real = split_im(real_color * mask, chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss*2)
        # Buildings color
        mask = torch.where(GT_seg == 2, 1., 0.)
        if torch.sum(mask) > 0:
            fake = split_im(fake_im * mask, chunk_nb)
            real = split_im(real_color * mask, chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss*0.2)
        # Motorcycle color
        mask = torch.where(GT_seg == 17, 1., 0.)
        if torch.sum(mask) > 0:
            fake = split_im(fake_im * mask, chunk_nb)
            real = split_im(real_color * mask, chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss)
        # Traffic sign color
        mask = torch.where(GT_seg == 7, 1., 0.)
        if torch.sum(mask) > 0:
            fake = split_im(fake_im * mask, chunk_nb)
            real = split_im(real_color * mask, chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss)
        # Road color
        mask = torch.where(GT_seg == 2, 1., 0.)
        if torch.sum(mask) > 0:
            fake = fake_im * mask
            real = real_color * mask
            fake = split_im(fake * (fake > fake.mean()*1.25), chunk_nb)
            real = split_im(real * (real > real.mean()*1.25), chunk_nb)
            loss = 0.
            for i in range(chunk_nb**2):
                if torch.sum(fake[:, :, i]) > 0:
                    hist_fake = hist_block(fake[:, :, i])
                    hist_real = hist_block(real[:, :, i])
                    loss += histogram_loss(hist_real, hist_fake)
            losses.append(loss)
    return sum(losses)


def ColorLoss(image_fake, image_target, GT_seg, chroma=False):
    im_fake = ImageTensor(image_fake*0.5 + 0.5).LAB()
    im_target = ImageTensor(image_target*0.5 + 0.5).LAB()
    color_fake = im_fake[0, 1:]
    color_target = im_target[0, 1:]
    colorAngle_fake = torch.arctan(color_fake[1]/color_fake[0])
    colorAngle_real = torch.arctan(color_target[1]/color_target[0])
    if chroma:
        Chroma_fake = torch.sqrt(color_fake[1]**2 + color_fake[0]**2)
        Chroma_real = torch.sqrt(color_target[1]**2 + color_target[0]**2)
        chroma_loss = torch.sqrt(torch.sum((Chroma_fake - Chroma_real) ** 2) + 1e-6) / color_fake.shape[-1]**2
    else:
        chroma_loss = 0
    loss_color = torch.sqrt(torch.sum(torch.cos(colorAngle_fake - colorAngle_real) ** 2) + 1e-6) / color_fake.shape[-1]**2 + chroma_loss


    if GT_seg is not None:
        ssim = SSIM_Loss(win_size=11, data_range=1.0, size_average=True, channel=1)
        i_fake = im_fake[0, 0]
        i_target = im_target[0, 0]
        losses = [0.]
        intensity_loss = lambda x, y, m: torch.sqrt((x*m - y*m)**2 + 1e-6) + ssim(x[None]*m[None], y[None]*m[None])
        tot_mask = 0
        # Cars color
        mask = torch.where(GT_seg == 13, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask)
        # Trucks color
        mask = torch.where(GT_seg == 14, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask)
        # Buildings color
        mask = torch.where(GT_seg == 2, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask)
        # Motorcycle color
        mask = torch.where(GT_seg == 17, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask)
        # Traffic sign color
        mask = torch.where(GT_seg == 7, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask*5)
        # Road color
        mask = torch.where(GT_seg <= 2, 1., 0.)
        sum_mask = torch.sum(mask)
        tot_mask += sum_mask
        if sum_mask > 0:
            loss = intensity_loss(i_fake, i_target, mask)
            losses.append((loss - loss.min()).sum()/sum_mask)
        loss_color += sum(losses) * tot_mask / color_fake.shape[-1]**2
    return loss_color












