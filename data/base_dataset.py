import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from .augmentation import TransResize, TransCrop, TransFlip, TransResize3, TransCrop3, TransFlip3, TransResizeN, \
    TransCropN, TransFlipN


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def night_train_transform(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(TransResize(opt.loadSize))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(TransCrop(opt.fineSize))
        if not opt.no_flip:
            transform_list.append(TransFlip(prob=0.5))

    transform_list.append(transforms.ToTensor())
    return transform_list

def night_train_transformv3(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(TransResize3(opt.loadSize))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(TransCrop3(opt.fineSize))
        if not opt.no_flip:
            transform_list.append(TransFlip3(prob=0.5))

    # transform_list.append(transforms.ToTensor())
    return transform_list

def night_train_transformvN(opt):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        transform_list.append(TransResizeN(opt.loadSize))

    if opt.isTrain:
        if 'crop' in opt.resize_or_crop:
            transform_list.append(TransCropN(opt.fineSize))
        if not opt.no_flip:
            transform_list.append(TransFlipN(prob=0.5))

    # transform_list.append(transforms.ToTensor())
    return transform_list