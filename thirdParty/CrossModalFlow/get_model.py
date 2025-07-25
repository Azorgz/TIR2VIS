import os

import torch
from .models.cross_raft import CrossRAFT

def get_model():
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(os.getcwd() + '/thirdParty/CrossModalFlow/cross_raft_ckpt/model/checkpoint-10000.ckpt', weights_only=True)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    return model