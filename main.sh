#!/bin/bash
#OAR -n tir2vis
#OAR -l /nodes=1/gpu=1,walltime=10:00:00
#OAR -p gpumodel='V100'
#OAR --stdout logs/print.out
#OAR --stderr logs/error.err
#OAR --project pr-remote-sensing-1a

python ~/TIR2VIS/train.py --continue_train True --which_epoch 8 --epoch_load 'latest'