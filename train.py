import torch
import time

from tqdm import tqdm
import socket
from models.combogan_model import GanColorCombo
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from util.visualizer import Visualizer

torch.set_num_threads(1)

opt = TrainOptions().parse()
opt.dataroot = '/silenus/PROJECTS/pr-remote-sensing-1a/godeta/FLIR/' if not 'laptop' in socket.gethostname() else opt.dataroot
opt.checkpoints_dir = '/bettik/PROJECTS/pr-remote-sensing-1a/godeta/checkpoints/' if not 'laptop' in socket.gethostname() else opt.checkpoints_dir
dataset = DataLoader(opt)
print('# training images = %d' % len(dataset))
model = GanColorCombo(opt)
visualizer = Visualizer(opt)
total_steps = 0
dataset_size = len(dataset) // opt.batchSize

# Update initially if continuing
model.update_hyperparams(opt.which_epoch)

for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    bar = tqdm(dataset, total=dataset_size, desc=f"epoch : {epoch}, step : 0, loss : 0")
    for i, data in enumerate(bar):
        iter_start_time = time.time()

        model.set_input(data)
        model.optimize_parameters(epoch)
        errors = model.get_current_errors()
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        bar.set_description(f"epoch : {epoch}, step : {i}, loss_G : {errors['G']}, loss_D : {errors['D']}")
        torch.cuda.empty_cache()

        if i % opt.save_step_latest == 0 and i !=0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_hyperparams(epoch)
