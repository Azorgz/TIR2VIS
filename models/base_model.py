import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch, gpu_ids):
        save_filename = f'{epoch}_net_{network_label}'
        save_path = os.path.join(self.save_dir, save_filename)
        network.save(save_path)
        if gpu_ids and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch):
        if isinstance(epoch, (str, int)):
            save_filename = f'{epoch}_net_{network_label}'
            save_path = os.path.join(self.save_dir, save_filename)
        else:
            save_filename = [f'{e}_net_{network_label}' for net, e in epoch.items() if network_label in net]
            save_path = [os.path.join(self.save_dir, fn) for fn in save_filename]
        network.load(save_path)

    def update_learning_rate(self, lr):
        self.lr = lr
