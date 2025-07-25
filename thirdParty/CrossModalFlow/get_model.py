import os
import socket

import torch
from .models.cross_raft import CrossRAFT

def get_model():
    model = CrossRAFT(adapter=True)
    path = os.getcwd() + '/thirdParty/CrossModalFlow/cross_raft_ckpt/model/checkpoint-10000.ckpt' if 'laptop' in socket.gethostname() else './bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/FoalGAN_FLIR/flow/checkpoint-10000.ckpt'
    state_dict = torch.load(path, weights_only=True)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    return model