import os.path, glob
import socket

import numpy as np
import oyaml
import torchvision.transforms as transforms
from einops import repeat
from kornia.filters import guided_blur

from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import SSIM
from data.base_dataset import BaseDataset, get_transform, night_train_transformvN
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import json


class EBGCDataset(BaseDataset):
    def __init__(self, opt):
        super(EBGCDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        self.night_edge_transform = night_train_transformvN(opt)
        if self.opt.isTrain:
            datapath = os.path.join(opt.dataroot, 'FLIR_datasets', opt.phase + '*')
            self.dirs = sorted(glob.glob(datapath))
            partial = 2
            self.IR_edge_paths = os.path.join(opt.dataroot, opt.IR_edge_path)  #######Edit by lfy########
            self.Vis_edge_paths = os.path.join(opt.dataroot, opt.Vis_edge_path)
            self.Vis_mask_paths = os.path.join(opt.dataroot, opt.Vis_mask_path)
            self.IR_mask_paths = os.path.join(opt.dataroot, opt.IR_mask_path)
            self.IR_FG_txt = os.path.join(opt.dataroot, opt.IR_FG_txt)
            self.Vis_FG_txt = os.path.join(opt.dataroot, opt.Vis_FG_txt)
            self.FB_Sample_Vis_txt = os.path.join(opt.dataroot, opt.FB_Sample_Vis_txt)
            self.FB_Sample_IR_txt = os.path.join(opt.dataroot, opt.FB_Sample_IR_txt)
            self.num_class = opt.num_class
            crop_path = '/silenus/PROJECTS/pr-remote-sensing-1a/godeta/FLIR/FLIR_datasets/crop.yaml' if not 'laptop' in socket.gethostname() else '/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/crop.yaml'
            with open(crop_path, 'r') as f:
                self.crop_xxyy = oyaml.safe_load(f)['crop']
        else:
            datapath = [opt.dataroot]
            self.dirs = datapath
            partial = opt.partial
        self.paths = [sorted(make_dataset(d, partial)) for d in self.dirs]  #####return all paths in three folders
        self.sizes = [len(p) for p in self.paths]  ####return the image number in two folders

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def load_image_train(self, dom, idx, sup_dom=None, crop=False):
        if crop:
            crop_xxyy = self.crop_xxyy[idx]
        else:
            crop_xxyy = [0, 0, 0, 0]
        path = self.paths[dom][idx]
        img = ImageTensor(path).crop(crop_xxyy, mode='lrtb').RGB()
        img_name = path.split('/')[-1]

        if sup_dom is not None:
            img_sup = ImageTensor(self.paths[sup_dom][idx]).RGB().crop(crop_xxyy, mode='lrtb')
        if dom == 0:
            edge_map = ImageTensor(self.Vis_edge_paths + img_name).GRAY().crop(crop_xxyy, mode='lrtb')
            seg_mask = (ImageTensor(self.Vis_mask_paths + img_name.replace('.jpg', '.png')).crop(crop_xxyy, mode='lrtb').match_shape(img, mode='nearest') * 255).to(torch.uint8)
        else:
            edge_map = ImageTensor(self.IR_edge_paths + img_name).GRAY().crop(crop_xxyy, mode='lrtb')
            seg_mask = (ImageTensor(self.IR_mask_paths + img_name.replace('.jpeg', '.png')).crop(crop_xxyy, mode='lrtb').match_shape(img, mode='nearest') * 255).to(torch.uint8)

        for func in self.night_edge_transform:
            if sup_dom is not None:
                img, edge_map, seg_mask, img_sup = func(img, edge_map, seg_mask, img_sup)
            else:
                img, edge_map, seg_mask = func(img, edge_map, seg_mask)
        transform_list_torch = [ImageTensor.to_tensor, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list_torch)
        if sup_dom is not None:
            img_fus = transform(self.fus_input(img_sup, img))
            # img_fus = transform(self.fus_input(img_sup, img))
            img_res = transform(img)
            img_sup_res = transform(img_sup)
            return img_res.squeeze(0), img_sup_res.squeeze(0), img_fus.squeeze(0), edge_map.squeeze(0), (seg_mask.squeeze() * 255).to(torch.uint8).clone()

        img_res = transform(img)
        return img_res.squeeze(0), edge_map.squeeze(0), (seg_mask.squeeze() * 255).to(torch.uint8).clone()

    def fus_input(self, img1, img2):
        ssim = SSIM(torch.device('cpu'))
        lab = ImageTensor(img1).LAB()
        image_target = lab[:, :1]
        image_ref = ImageTensor(img2).RGB('gray')
        ssim_mask = ssim(image_target, image_ref, return_image=True)
        ssim_mask = ssim_mask.GRAY().resize(image_target.shape[-2:])
        ssim_mask = guided_blur(image_ref, ssim_mask, 5, 0.1)
        ssim_fused_L = lab[:, :1] * ssim_mask + (1 - ssim_mask) * img2.GRAY()
        # ssim_fused_L = img2.GRAY()
        lab = self.color_clamp(lab, ssim_mask)
        lab[:, :1] = ssim_fused_L
        return lab.RGB().detach()

    def color_clamp(self, lab, mask, ratio=5):
        L, AB = lab[:, :1], lab[:, 1:]
        # AB = (AB - 0.5) * (1+ratio/100) + 0.5 * (1+ratio/100)
        # AB = torch.clamp(AB, 0, 1)
        # mask = torch.where((L < L.mean() - L.std()) * (L > L.mean() + 3 * L.std()), 0, 1) * mask
        mask_AB = repeat(mask, 'b () h w -> b c h w', c=2)
        AB = AB * (mask_AB > 0.5) + 0.5 * (mask_AB <= 0.5)
        lab[:, 1:] = AB
        return lab

    def load_image_train_crop(self, dom, idx, crop_pos_h, crop_pos_w):
        path = self.paths[dom][idx]
        img = np.array(Image.open(path).convert('RGB'))
        path_split = path.split('/')
        img_name = path_split[-1]
        if dom == 0:
            edge_map_file = self.Vis_edge_paths + img_name
            seg_mask_file = self.Vis_mask_paths + img_name
            seg_mask = np.array(Image.open(seg_mask_file))
        else:
            edge_map_file = self.IR_edge_paths + img_name
            seg_mask_file = self.IR_mask_paths + img_name
            seg_mask = np.array(Image.open(seg_mask_file))

        edge_map = np.array(Image.open(edge_map_file).convert('L'))

        w1 = crop_pos_h
        h1 = crop_pos_w

        img_crop = img[w1 : (w1 + 256), h1 : (h1 + 256)]
        edge_map_crop = edge_map[w1 : (w1 + 256), h1 : (h1 + 256)]
        seg_mask_crop = seg_mask[w1 : (w1 + 256), h1 : (h1 + 256)]

        if np.random.rand() < 0.5:
            img_crop = img_crop[:,::-1]
            edge_map_crop = edge_map_crop[:,::-1]
            seg_mask_crop = seg_mask_crop[:,::-1]

        transform_list_torch = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_new = transforms.Compose(transform_list_torch)
        # img = torch.from_numpy(img_file.transpose((2, 0, 1)))
        img_new = Image.fromarray(np.uint8(img_crop))
        img_res = transform_new(img_new)
        # print(img_res.size)
        edge_map_crop = edge_map_crop / 255.0
        seg_mask_crop = np.asarray(Image.fromarray(seg_mask_crop), dtype=np.int64)

        return img_res, torch.tensor(edge_map_crop), seg_mask_crop.copy()


    def count_class_ratio(self, input_mask):
        count = np.zeros((1, self.num_class))
        h, w = input_mask.shape
        for i in range(self.num_class):
            count[0, i] = (np.sum(input_mask == i)) / (h * w)
        ###Small sample categories includes six categories: traffic light, traffic sign, person, truck, bus, and motorcycle. 
        ###Their indexes are 6, 7, 11, 14, 15 and 17, respectively.
        SSC_idx = np.zeros((self.num_class, 1))
        SSC_idx[6:8, 0] = 1.0
        SSC_idx[11, 0] = 1.0
        SSC_idx[14:16, 0] = 1.0
        SSC_idx[17, 0] = 1.0
        SSC_ratio_sum = np.dot(count, SSC_idx)
        if SSC_ratio_sum == 0.0:
            cls_ratio_norm = np.zeros((1, 6))
        else:
            SSC_ratio = np.zeros((1, 6))
            SSC_ratio[0, 0:2] = count[0, 6:8]
            SSC_ratio[0, 2] = count[0, 11]
            SSC_ratio[0, 3:5] = count[0, 14:16]
            SSC_ratio[0, 5] = count[0, 17]
            count_submean = SSC_ratio - (SSC_ratio_sum / 6.0)
            cls_ratio_norm = count_submean / np.linalg.norm(count_submean, axis=1, keepdims=True)
        # print(cls_ratio_norm)

        return cls_ratio_norm, SSC_ratio_sum

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d, s in enumerate(self.sizes):
                    if index < s:
                        DA = d;
                        break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)

            A_img, A_path = self.load_image(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path}
        else:
            # Choose two of our domains to perform a pass on
            # DA, DB = random.sample(range(len(self.dirs)), 2) #########
            DA, DB, DC, DFus = 0, 1, 2, 3
            with open(self.FB_Sample_Vis_txt, 'r') as FGsampVis:
                if 'True' in FGsampVis.read():
                    # print('DataLoader sampling FG is True.')
                    A_img0_path = self.paths[DA][0]
                    pathA_split = A_img0_path.split('/')
                    A_img0_name = pathA_split[-1]
                    pathA = A_img0_path.replace(A_img0_name, '')
                    with open(self.Vis_FG_txt, 'r') as VisFGList:
                        lines_FG_Vis = VisFGList.readlines()
                        index_list_A = random.randint(0, len(lines_FG_Vis) - 1)
                        line_txt = lines_FG_Vis[index_list_A].strip('\n')
                        line_content_split = line_txt.split(' ')
                        temp_img_path = pathA + line_content_split[0]
                        pos_h = int(line_content_split[1]) - 1
                        pos_w = int(line_content_split[2]) - 1
                        index_A = self.paths[DA].index(temp_img_path)

                    A_img, edge_map_A, seg_mask_A = self.load_image_train_crop(DA, index_A, pos_h, pos_w)
                    bundle = {'A': A_img, 'DA': DA, 'EMA': edge_map_A, 'SMA': seg_mask_A}

                else:
                    index_A = random.randint(0, self.sizes[DA] - 1)
                    A_img, edge_map_A, seg_mask_A = self.load_image_train(DA, index_A)
                    bundle = {'A': A_img, 'DA': DA, 'EMA': edge_map_A, 'SMA': seg_mask_A}

                with open(self.FB_Sample_IR_txt, 'r') as FGsampIR:
                    if 'True' in FGsampIR.read():
                        # print('DataLoader sampling FG is True.')
                        B_img0_path = self.paths[DB][0]
                        pathB_split = B_img0_path.split('/')
                        B_img0_name = pathB_split[-1]
                        pathB = B_img0_path.replace(B_img0_name, '')
                        with open(self.IR_FG_txt, 'r') as IRFGList:
                            lines_FG_IR = IRFGList.readlines()
                            index_list_B = random.randint(0, len(lines_FG_IR) - 1)
                            line_txt_B = lines_FG_IR[index_list_B].strip('\n')
                            line_content_split_B = line_txt_B.split(' ')
                            temp_img_path_B = pathB + line_content_split_B[0]
                            pos_h_B = int(line_content_split_B[1]) - 1
                            pos_w_B = int(line_content_split_B[2]) - 1
                            index_B = self.paths[DB].index(temp_img_path_B)

                        B_img, edge_map_B, seg_mask_B = self.load_image_train_crop(DB, index_B, pos_h_B, pos_w_B)
                        C_img, edge_map_B, seg_mask_B = self.load_image_train_crop(DC, index_B, pos_h_B, pos_w_B)
                        # bundle = {'B': B_img, 'DB': DB, 'path': B_path, 'EMB':edge_map_B, 'SMB':seg_mask_B}

                    else:
                        index_B = random.randint(0, self.sizes[DB] - 1)
                        B_img, C_img, Fus_img, edge_map_B, seg_mask_B = self.load_image_train(DB, index_B, sup_dom=DC, crop=True)
                        # bundle = {'B': B_img, 'DB': DB, 'path': B_path, 'EMB':edge_map_B, 'SMB':seg_mask_B}
                    # C_img = self.load_image(DC, index_B)[0]

                bundle.update({'B': B_img, 'DB': DB, 'Fus': Fus_img,
                               'EMB': edge_map_B, 'SMB': seg_mask_B, 'C': C_img, 'DC': DC, 'DFus': DFus})

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'EBGCDataset'
